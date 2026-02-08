from itertools import groupby
from typing import Any, List

import torch
from einops import rearrange
from torch import nn

from .layers import MarkerAttentionEncoderBlock, SA_BIAS_CACHE


def build_multiplex_tokenizer(conf, protein_emb, **kwargs) -> nn.Module:
    tokenizer = MultiplexTokenizer(
        protein_emb=protein_emb,
        patch_size=conf.image_info.patch_size,
        model_dim=conf.model.model_dim,
        feedforward_dim=conf.model.feedforward_dim,
        encoder_pattern=conf.model.encoder_pattern,
        num_encoder_heads=conf.model.num_encoder_heads,
        dropout=conf.model.dropout,
        group_layers=conf.model.group_layers,
        **kwargs,
    )
    return tokenizer


class MultiplexTokenizer(nn.Module):
    def __init__(
        self,
        protein_emb: torch.Tensor,
        patch_size: int,
        model_dim: int,
        feedforward_dim: int,
        encoder_pattern: str,
        num_encoder_heads: int,
        dropout: float,
        group_layers: bool,
        **kwargs: Any,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        self.encoder_pattern = encoder_pattern
        self.num_encoder_heads = num_encoder_heads
        self.dropout = dropout

        self.register_buffer("protein_emb", protein_emb, persistent=False)
        self.protein_encoder = nn.Linear(protein_emb.shape[1], model_dim)

        # model specific
        power = kwargs.get("parameter_init_power", 0.5)
        self.patch_summary_token = nn.Parameter(torch.randn(self.model_dim) / self.model_dim**power)  # type: ignore

        self.multiplex_patch_encoder = nn.Linear(self.patch_size * self.patch_size, self.model_dim)
        # forming encoder
        enc_layers = []
        if group_layers:
            grouped_encoder_pattern = [(label, sum(1 for _ in group)) for label, group in groupby(self.encoder_pattern)]
        else:
            grouped_encoder_pattern = [(label, 1) for label in self.encoder_pattern]
        for block_type, count in grouped_encoder_pattern:
            if block_type == "v":
                enc_layers.append(
                    MarkerAttentionEncoderBlock(
                        model_dim=self.model_dim,
                        feedforward_dim=self.feedforward_dim,
                        num_heads=self.num_encoder_heads,
                        dropout=dropout,
                        inbuilt_pos_emb=None,
                        num_layers=count,
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported block type {block_type} for the multiplex tokenizer.")

        self.layer_norm = nn.LayerNorm(self.model_dim)
        self.encoder = nn.ModuleList(enc_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, multiplex: List[torch.Tensor], channel_ids: torch.Tensor):
        return self.forward_list(multiplex, channel_ids)

    def forward_list(self, multiplex: List[torch.Tensor], channel_ids: List[torch.Tensor]):
        h, w = multiplex[0].shape[-2], multiplex[0].shape[-1]
        H, W = h // self.patch_size, w // self.patch_size
        B = len(multiplex)

        multiplex_channels_per_sample = [len(ch) for ch in channel_ids]
        multiplex = [(rearrange(mx_i, "C (H p) (W q) -> C H W (p q)", p=self.patch_size, q=self.patch_size)) for mx_i in multiplex]
        cat_multiplex = torch.cat(multiplex, dim=0)  # (sum_C) H W (p q)
        cat_channel_ids = torch.cat(channel_ids, dim=0).to(self.protein_emb.device)  # (sum_C)
        sum_C = sum(multiplex_channels_per_sample) + B * 1  # the patch summary tokens

        cat_multiplex = self.multiplex_patch_encoder(cat_multiplex)  # (sum_C) (H W) model_dim
        cat_multiplex = rearrange(cat_multiplex, "C H W D -> C (H W) D")

        protein_emb = self.protein_emb[cat_channel_ids].to(dtype=torch.float16)  # (sum_C) D
        protein_emb = self.protein_encoder(protein_emb)  # (sum_C) model_dim
        protein_emb = protein_emb.unsqueeze(1).expand(*cat_multiplex.shape)  # (sum_C) (H W) model_dim
        cat_multiplex = cat_multiplex + protein_emb

        pos_multiplex = torch.stack(
            torch.meshgrid(torch.arange(H, device=cat_multiplex.device), torch.arange(W, device=cat_multiplex.device), indexing="ij"), dim=-1
        )
        pos_multiplex = pos_multiplex.expand(sum_C, H, W, 2)
        pos_multiplex = rearrange(pos_multiplex, "C H W d -> C (H W) d")

        cat_multiplex = torch.split(cat_multiplex, multiplex_channels_per_sample, dim=0)  # List[(C_i) (H W) D]
        x = [
            torch.cat(
                [
                    self.patch_summary_token.expand(1, H * W, self.patch_summary_token.shape[-1]),
                    mx_i,
                ],
                dim=0,
            )
            for mx_i in cat_multiplex
        ]
        x = torch.cat(x, dim=0)  # (sum_C + 1) (H W) D
        x_channels_per_sample = [mc + 1 for mc in multiplex_channels_per_sample]

        for layer in self.encoder:
            layer_output = layer.forward_cc(x, pos_multiplex, x_channels_per_sample)  # type: ignore
            x = layer_output

        x = self.layer_norm(x)

        SA_BIAS_CACHE.clear()  # clear cache after forward pass
        x = rearrange(x, "C (H W) D -> C H W D", H=H, W=W)
        x = torch.split(x, x_channels_per_sample, dim=0)  # List[(C_i + 1) H W D]
        ps = [x_i[0] for x_i in x]  # List[(H W) D]
        x = [x_i[1:] for x_i in x]  # List[(C_i) H W D]

        return ps
