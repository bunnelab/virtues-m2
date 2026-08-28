from itertools import groupby
from typing import List, Optional

import torch
from einops import rearrange
from torch import nn

from virtuesm2.models.layers import FullAttentionEncoderBlock, SA_BIAS_CACHE

import math
import torch.nn.functional as Fn


class UpsampleHead(nn.Module):
    """Patch-agnostic: upsample a token grid to an explicit target resolution.

    Two changes from the bilinear-only version:
      * learned upsampling via PixelShuffle (sub-pixel conv) instead of fixed
        bilinear interpolation -- this can synthesize high frequency, bilinear
        cannot. A single bilinear "snap" remains only to hit a non-power-of-2
        target exactly (e.g. 32x overshoot -> 24x target).
      * an optional raw-H&E guide fused near target resolution. The fusion conv
        is zero-initialised, so at step 0 the head is numerically identical to a
        no-guide head and the guide enters as a learned residual (safe to resume
        from an existing checkpoint; watch the fuse weights grow if it helps).
    """

    def __init__(self, model_dim, base=128, out_ch=1, n_blocks=4, guide_ch=0):
        super().__init__()
        self.guide_ch = guide_ch

        self.proj = nn.Conv2d(model_dim, base, 1)

        # learned 2x stages (no bilinear in the main path)
        self.up = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base, base * 4, 3, padding=1),
                nn.PixelShuffle(2),                       # (base*4, h, w) -> (base, 2h, 2w)
                nn.GroupNorm(min(8, base), base), nn.GELU(),
                nn.Conv2d(base, base, 3, padding=1),
                nn.GroupNorm(min(8, base), base), nn.GELU(),
            )
            for _ in range(n_blocks)
        ])

        # raw-H&E guide branch (encoded once per *sample*, then expanded over channels)
        if guide_ch > 0:
            self.guide = nn.Sequential(
                nn.Conv2d(guide_ch, base, 3, padding=1), nn.GELU(),
                nn.Conv2d(base, base, 3, padding=1),
            )
            self.fuse = nn.Conv2d(base * 2, base, 1)
            nn.init.zeros_(self.fuse.weight)              # zero-init: starts as a no-op,
            nn.init.zeros_(self.fuse.bias)                # learns the H&E blend

        self.out = nn.Conv2d(base, out_ch, 1)
        nn.init.xavier_normal_(self.out.weight, gain=0.1)
        nn.init.zeros_(self.out.bias)

    def forward(self, tok, target_hw, guide=None, repeats=None):
        """
        tok     : (sum_C, model_dim, H, W)   token grid, rows grouped by sample
        guide   : (N, guide_ch, Hpx, Wpx)    raw (stain-normalized) H&E, one per sample,
                                             same order as `repeats`. None disables fusion.
        repeats : list[int] of length N -- channels per sample, so guide features can be
                                             expanded to match tok's row grouping.
        """
        th, tw = target_hw
        x = self.proj(tok)

        for blk in self.up:
            if x.shape[-2] >= th and x.shape[-1] >= tw:   # stop once we've reached/passed target
                break
            x = blk(x)

        if x.shape[-2:] != (th, tw):                      # snap to exact (handles the 24x remainder)
            x = Fn.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)

        if self.guide_ch > 0 and guide is not None:
            g = guide
            if g.shape[-2:] != (th, tw):                  # ideally a no-op/downsample, never an upsample
                g = Fn.interpolate(g, size=(th, tw), mode="bilinear", align_corners=False)
            g = self.guide(g)                             # (N, base, th, tw) -- N convs, not sum_C
            if repeats is not None:
                g = torch.cat([g[b:b + 1].expand(repeats[b], -1, -1, -1)
                               for b in range(len(repeats))], dim=0)  # (sum_C, base, th, tw)
            x = x + self.fuse(torch.cat([x, g], dim=1))   # gated raw-H&E fusion (zero at init)

        return self.out(x)


class VirtualStainingDecoder(nn.Module):
    def __init__(
        self,
        patch_size=24,
        model_dim=512,
        feedforward_dim=1024,
        pattern="f",
        num_heads=8,
        num_hidden_layers_head=0,
        dropout=0.0,
        pos_emb="rope",
        separate_decoders=False,
        group_layers=False,
        norm_after_encoder_decoder=False,
        conv_upsample_features: int = 0,
        skip_layers: int = 0,
        he_guide_channels: int = 3,        # NEW: channels of the raw H&E guide (0 disables)
        **kwargs,
    ):
        super().__init__()

        self.separate_decoders = separate_decoders
        self.norm_after_encoder_decoder = norm_after_encoder_decoder
        self.patch_size = patch_size
        self.conv_upsample_features = conv_upsample_features
        self.skip_layers = skip_layers
        self.he_guide_channels = he_guide_channels

        if skip_layers > 0:
            self.skip_proj = nn.ModuleList([
                nn.Linear(model_dim, model_dim) for _ in range(skip_layers)
            ])

        self.cross_channel = kwargs.get("cross_channel", False)   # NEW

        if conv_upsample_features > 0:
            self.conv_upsample_head = UpsampleHead(
                model_dim=model_dim,
                base=conv_upsample_features,
                out_ch=1,                                   # no raw-input skip; predict from ps only
                n_blocks=int(round(math.log2(patch_size))),
                guide_ch=he_guide_channels,                 # raw H&E enters here
            )
        else:
            multiplex_decoder_layers = []
            if num_hidden_layers_head > 0:
                for _ in range(num_hidden_layers_head - 1):
                    multiplex_decoder_layers.append(nn.Linear(model_dim, model_dim))
                    multiplex_decoder_layers.append(nn.GELU())
            multiplex_decoder_layers.append(nn.Linear(model_dim, patch_size**2))
            self.multiplex_decoder_mlp = nn.Sequential(*multiplex_decoder_layers)

        if group_layers:
            groups = groupby(pattern)
            pattern = [(label, sum(1 for _ in group)) for label, group in groups]
        else:
            pattern = [(label, 1) for label in pattern]

        dec_layers = []
        for pattern, depth in pattern:
            if pattern == "f":
                dec_layers.append(
                    FullAttentionEncoderBlock(
                        model_dim, num_heads, feedforward_dim,
                        dropout=dropout, inbuilt_pos_emb=pos_emb, num_layers=depth,
                    )
                )
            else:
                raise ValueError(
                    f"decoder_pattern '{pattern}' not supported. "
                    "Only 'f' (full attention) is currently supported for the decoder."
                )
        self.decoder = nn.ModuleList(dec_layers)

        if norm_after_encoder_decoder:
            self.layer_norm = nn.LayerNorm(model_dim)

        if skip_layers > 0:
            self.head_skip_proj = nn.ModuleList([
                nn.Linear(model_dim, model_dim) for _ in range(skip_layers)
            ])

    def forward(self, x, ps, multiplex_channels_per_sample,
                target_hw, he_full=None, skip_features=None):
        return self.forward_list(
            x, ps, multiplex_channels_per_sample,
            target_hw, he_full=he_full, skip_features=skip_features,
        )

    def forward_list(self, x, ps, multiplex_channels_per_sample,
                     target_hw, he_full=None, skip_features=None):
        """
        he_full : (N, he_guide_channels, Hpx, Wpx) raw H&E, one image per *sample*, in the
                  same order as `ps` / `multiplex_channels_per_sample`. Should be registered
                  to the multiplex GT and stain-normalized. Pass None to disable the guide.
        """
        H, W, D = x[0].shape[1], x[0].shape[2], x[0].shape[3]
        x_channels_per_sample = multiplex_channels_per_sample

        if skip_features is not None:
            skip_features = list(reversed(skip_features))

        parts = []
        for x_i, ps_i in zip(x, ps):
            c = x_i.shape[0]
            parts.append(torch.stack([ps_i.expand(c, H, W, D), x_i], dim=1))  # (C_i, 2, H, W, D)
        x = torch.cat(parts, dim=0)                              # (sum_C, 2, H, W, D)
        x = rearrange(x, "c e h w d -> (c e) (h w) d")           # (sum_C*2, S, D)

        pos = torch.stack(torch.meshgrid(
            torch.arange(H, device=x.device), torch.arange(W, device=x.device),
            indexing="ij"), dim=-1)
        pos = rearrange(pos.expand(x.shape[0], H, W, 2), "c h w d -> c (h w) d")
        examples = x.shape[0] // 2

        # per-channel blocks (independent) vs per-sample blocks (cross-channel)
        if self.cross_channel:
            groups = [2 * c for c in multiplex_channels_per_sample]
        else:
            groups = [2] * examples

        for i, layer in enumerate(self.decoder):
            if skip_features is not None and i < len(skip_features):
                sf = skip_features[i]
                sf_exp = torch.cat([
                    sf[b].unsqueeze(0).expand(multiplex_channels_per_sample[b], H, W, D)
                    for b in range(sf.shape[0])], dim=0)
                delta = torch.zeros_like(x)
                delta[0::2] = self.skip_proj[i](rearrange(sf_exp, "c h w d -> c (h w) d"))
                x = x + delta
            x = layer.forward_cc(x, pos, groups)

        if self.norm_after_encoder_decoder:
            x = self.layer_norm(x)

        x = x[1::2]                                              # query tokens
        x = torch.split(x, x_channels_per_sample, dim=0)
        multiplex = torch.concat([xi[:c] for xi, c in zip(x, multiplex_channels_per_sample)], dim=0)

        # raw per-position residual (final-layer encoder features)
        raw = torch.cat([ps_i.reshape(H * W, D).unsqueeze(0).expand(c, H * W, D)
                         for ps_i, c in zip(ps, multiplex_channels_per_sample)], dim=0)
        multiplex = multiplex + raw

        # fuse intermediate encoder features directly into the head (patch-res; both head types)
        if skip_features is not None and self.skip_layers > 0:
            for k, sf in enumerate(skip_features):
                sf_pc = torch.cat([
                    sf[b].reshape(H * W, D).unsqueeze(0).expand(
                        multiplex_channels_per_sample[b], H * W, D)
                    for b in range(len(multiplex_channels_per_sample))], dim=0)   # (sum_C, S, D)
                multiplex = multiplex + self.head_skip_proj[k](sf_pc)

        SA_BIAS_CACHE.clear()

        if self.conv_upsample_features > 0:
            tok = rearrange(multiplex, "c (h w) d -> c d h w", h=H, w=W)
            multiplex = self.conv_upsample_head(
                tok, target_hw,
                guide=he_full,                                  # raw H&E -> high-freq detail
                repeats=multiplex_channels_per_sample,          # expand guide rows to match tok
            ).squeeze(1)
        else:
            raise ValueError("MLP head bakes in patch_size**2 — not shareable across patch sizes; "
                             "use the conv head for the shared decoder")

        multiplex = torch.split(multiplex, multiplex_channels_per_sample, dim=0)
        return [m if m.shape[0] > 0 else None for m in multiplex]