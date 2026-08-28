from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from virtuesm2.models.virtuesm2 import VirTuesM2
from virtuesm2.utils.utils import (load_marker_embedding_dict,
                                   load_marker_embeddings)
from virtuesm2.virtual_staining.decoder import VirtualStainingDecoder
from virtuesm2.virtual_staining.positional_embeddings import \
    PositionalEmbedding2D


class VM2_VirtualStaining(nn.Module):
    """VirTues-M2 virtual staining model.

    Loads the frozen VirTues-M2 backbone from HuggingFace (bunnelab/virtues-m2)
    and trains a VirtualStainingDecoder on top to predict target MX channels
    from H&E patch tokens and ESM protein embeddings.

    Parameters
    ----------
    vm2_encoder:
        The pre-trained VirTuesM2 encoder.
    marker_embedding_dir:
        Directory containing the marker embeddings.
    mode:
        "he" for H&E to MX, "mx" for MX to H&E (default "he").
    """

    def __init__(
        self,
        vm2_encoder: VirTuesM2,
        marker_embedding_dir: str,
        mode: str = "he",
    ):
        super().__init__()
        self.uniprot_to_idx = load_marker_embedding_dict(marker_embedding_dir)
        self.esm_embeddings = load_marker_embeddings(marker_embedding_dir)
        self.mode = mode
        # self.uniprot_to_idx: Dict[str, int] = {uid: i for i, uid in enumerate(uniprot_ids)}
        self.model = vm2_encoder
        self.embed_dim = self.model.embed_dim
        self.skip_block_indices = [5, 11, 17]
        self.skip_layers = len(self.skip_block_indices)

        # Decoder
        self.decoder = VirtualStainingDecoder(
            patch_size=self.model.patch_size,
            model_dim=self.embed_dim,
            feedforward_dim=2048,
            pattern="ff",
            num_heads=8,
            norm_after_encoder_decoder=True,
            group_layers=True,
            conv_upsample_features=128,
            skip_layers=self.skip_layers,
        )

        self.protein_encoder = nn.Linear(self.esm_embeddings.shape[1], self.embed_dim).cuda()

        self.positional_embedding = PositionalEmbedding2D(
            model_dim=self.embed_dim, max_width_or_height=1200
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _to_protein_indices(self, uniprot_ids: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            [self.uniprot_to_idx[u] for u in uniprot_ids], dtype=torch.long
        )
    
    @torch.inference_mode()
    def forward(self, batch: List[Dict[str, Any]]) -> List[torch.Tensor]:
        if self.mode == "he":
            he_imgs = [s["he"].to(self.device) for s in batch]
            mx_imgs = [None] * len(batch)
            patch_key = "x_norm_patchtokens_he"
            h = batch[0]["he"].shape[-2] // self.model.patch_size
            w = batch[0]["he"].shape[-1] // self.model.patch_size
            input_protein_idx = [None] * len(batch)
        else:
            mx_imgs = [s["mx"].to(self.device) for s in batch]
            he_imgs = [None] * len(batch)
            patch_key = "x_norm_patchtokens_multiplex"
            h = batch[0]["mx"].shape[-2] // self.model.patch_size
            w = batch[0]["mx"].shape[-1] // self.model.patch_size
            input_protein_idx = [
                self._to_protein_indices(s["uniprot_ids"]).to(self.device)
                for s in batch
            ]

        mm_embeddings = self.model.forward_features(
            mx_imgs, he_imgs, input_protein_idx,
            return_all_layers=(self.skip_layers > 0)
        )
        if self.skip_layers > 0:
            n_reg = self.model.num_register_tokens
            skip_feats = []
            for idx in self.skip_block_indices:
                raw = mm_embeddings["x_all_layers"][idx]
                normed = self.model.norm(raw)  # (B, 1+n_reg+2*N, D)
                if self.mode == "he":
                    tok = normed[:, 1 + n_reg + h * w:]  # (B, N, D)
                else:
                    tok = normed[:, 1 + n_reg : 1 + n_reg + h * w]  # (B, N, D)
                skip_feats.append(rearrange(tok, "b (h w) d -> b h w d", h=h, w=w))
        else:
            skip_feats = None
        patch_tokens = mm_embeddings[patch_key]  # (B, N, D)

        ps = rearrange(patch_tokens, "b (h w) d -> b h w d", h=h, w=w)  # (B, h, w, D)

        target_protein_idx = [
            self._to_protein_indices(s["target_uniprot_ids"]).to(self.device)
            for s in batch
        ]

        n_targets = [len(idx) for idx in target_protein_idx]
        esm = self.esm_embeddings.to(self.device)
        d = self.embed_dim

        # Build per-sample query tokens: list of (C_i, h, w, D)
        x = []
        for prot_idx in target_protein_idx:
            mask_tok = self.model.mx_mask_token.expand(len(prot_idx), h, w, d).clone()
            sel = self.protein_encoder(esm[prot_idx])  # (C_i, D)
            mask_tok = mask_tok + sel.unsqueeze(1).unsqueeze(1).expand_as(mask_tok)
            pos = torch.stack(
                torch.meshgrid(
                    torch.arange(h, device=self.device),
                    torch.arange(w, device=self.device),
                    indexing="ij",
                ),
                dim=-1,
            ).unsqueeze(0).expand(len(prot_idx), -1, -1, -1)  # (C_i, h, w, 2)
            mask_tok = self.positional_embedding(mask_tok, pos)
            x.append(mask_tok)

        ps_list = [ps[i] for i in range(len(batch))]
        
        if self.mode == "he":
            target_hw = (he_imgs[0].shape[-2], he_imgs[0].shape[-1])
        else:
            target_hw = (mx_imgs[0].shape[-2], mx_imgs[0].shape[-1])
            
        predictions = self.decoder.forward_list(x, ps_list, n_targets, target_hw, skip_features=skip_feats)
        return [p for p in predictions if p is not None]
