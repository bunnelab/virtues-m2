"""Sinusoidal 2D positional embedding for channels-last tensors.

Accepts inputs of shape (..., H, W, D) and positions of shape (..., H, W, 2)
where positions[..., 0] is the row index and positions[..., 1] is the column
index for each spatial location.  This matches the convention used in the
virtual-staining decoder and the notebook inference code.
"""

import torch
import torch.nn as nn
from typing import Optional


class PositionalEmbedding2D(nn.Module):
    def __init__(
        self,
        model_dim: int,
        max_width_or_height: int = 1200,
        temperature: float = 10000.0,
    ) -> None:
        super().__init__()
        if model_dim % 4 != 0:
            raise ValueError("model_dim must be divisible by 4.")
        self.model_dim = model_dim
        self.max_width_or_height = max_width_or_height
        self.temperature = temperature
        self.register_buffer("positional_encoding", self._create_pe(), persistent=False)

    def _create_pe(self) -> torch.Tensor:
        """Returns (max_width_or_height, model_dim // 2)."""
        dim_pe = self.model_dim // 2
        positions = torch.arange(self.max_width_or_height, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_pe, 2).float()
            * (-torch.log(torch.tensor(self.temperature)) / dim_pe)
        )
        pe = torch.zeros(self.max_width_or_height, dim_pe)
        pos = positions * div_term
        pe[:, 0::2] = torch.sin(pos)
        pe[:, 1::2] = torch.cos(pos)
        return pe  # (max_width_or_height, dim_pe)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (..., H, W, D) channels-last tensor.
            positions: (..., H, W, 2) long tensor with (row, col) indices.
                       If None, a default grid [0..H-1] x [0..W-1] is used.
        Returns:
            x + positional embedding, same shape as x.
        """
        *batch, H, W, D = x.shape
        assert D == self.model_dim, f"Last dim {D} != model_dim {self.model_dim}"

        if positions is None:
            rows = torch.arange(H, device=x.device)
            cols = torch.arange(W, device=x.device)
            row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")  # (H, W)
            row_emb = self.positional_encoding[row_grid]   # (H, W, D//2)
            col_emb = self.positional_encoding[col_grid]   # (H, W, D//2)
        else:
            # positions: (..., H, W, 2)
            row_idx = positions[..., 0]  # (..., H, W)
            col_idx = positions[..., 1]  # (..., H, W)
            row_emb = self.positional_encoding[row_idx]    # (..., H, W, D//2)
            col_emb = self.positional_encoding[col_idx]    # (..., H, W, D//2)

        pe = torch.cat([row_emb, col_emb], dim=-1)  # (..., H, W, D)
        return x + pe.to(x.dtype)
