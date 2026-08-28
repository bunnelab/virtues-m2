from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from instanseg.utils.tiling import _chops, _tiles_from_chops, _stitch, _stitch_mean

CELL_TYPE_COLOR_MAP = {
    "Tumor":"#71D49F",
    "Myeloid": "#45020D",
    "B cell": "#972F3E",
    "CD4 T cell": "#E1A95F",
    "CD8 T cell": "#B87333",
    "Fibroblasts /Stroma": "#17256B",
    "Vessel / Endo.": "#A699B3",
}

FALLBACK = "#DDDDDD"  # classes with no assigned color

CELL_TYPE_MAPPING = {
    "None": 0,
    "B cell": 1,
    "CD4 T cell": 2,
    "CD8 T cell": 3,
    "Fibroblasts /Stroma": 4,
    "Myeloid": 5,
    "Tumor": 6,
    "Unknown": 7,
    "Vessel / Endo.": 8,
}

def remove_small_cells(prediction: torch.Tensor | np.ndarray, min_cell_size: int = 15) -> np.ndarray:
    labels, counts = np.unique(prediction, return_counts=True)
    small_cells = labels[counts < min_cell_size]
    prediction = np.where(np.isin(prediction, small_cells), 0, prediction)
    labels, inverse = np.unique(prediction, return_inverse=True)
    relabelled = inverse.reshape(prediction.shape)
    return relabelled.astype(np.int32)

def assign_cell_types(instance_prediction: np.ndarray, semantic_prediction: np.ndarray, num_classes: int = 9) -> np.ndarray:
    """Assign every pixel of each instance the most common per-pixel semantic prediction within that instance."""
    flat_inst = instance_prediction.ravel()
    flat_sem = semantic_prediction.ravel()
    valid = flat_inst != 0
    uniq_inst, inst_idx = np.unique(flat_inst[valid], return_inverse=True)
    combined = inst_idx * num_classes + flat_sem[valid]
    counts = np.bincount(combined, minlength=len(uniq_inst) * num_classes).reshape(len(uniq_inst), num_classes)
    majority_class = counts.argmax(axis=1)

    result = np.full(flat_inst.shape, fill_value=np.nan)
    result[valid] = majority_class[inst_idx]
    return result.reshape(instance_prediction.shape)

def segment_large_tissue(
    mx_tissue: torch.Tensor | None,
    he_tissue: torch.Tensor | None,
    segmentation_model: nn.Module,
    channel_ids: torch.Tensor | None,
    tile: int,
    ovlp: int,
    bs: int,
    *,
    max_seeds: int = 30000,
    window_size: int = 64,
    detection_size: int = 20,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Predict a full-tissue instance mask and phenotype logits for tissues too large to
    segment in one shot.

    Args:
        mx_tissue: Multiplexed input image of shape (C, H, W), or None if the decoder's mode does not
            use the multiplex branch.
        he_tissue: H&E input image of shape (3, H, W), or None if the decoder's mode does not use the
            H&E branch.
        segmentation_model: A VirTuesM2SegmentationDecoder with `instance_processor` and `dim_out` set.
        channel_ids: A tensor of shape (C,) containing the channel IDs for mx_tissue.
        tile: The size of the tiles to use for segmentation.
        ovlp: The amount of overlap between tiles.
        bs: The batch size to use for segmentation.
        max_seeds: The maximum number of seeds to use for instance segmentation. Defaults to 30000.
        window_size: The window size to use for instance segmentation. Defaults to 64.
        detection_size: The detection size to use for instance segmentation. Defaults to 20.
    Returns:
        A tuple containing the predicted instance mask and phenotype logits.
    """
    reference_tissue = mx_tissue if mx_tissue is not None else he_tissue
    h, w = int(reference_tissue.shape[-2]), int(reference_tissue.shape[-1])
    tile_hw = (min(tile, h), min(tile, w))
    chop_idx = _chops(reference_tissue.shape, shape=tile_hw, overlap=2 * (ovlp + detection_size))

    mx_tiles = _tiles_from_chops(mx_tissue, shape=tile_hw, tuple_index=chop_idx) if mx_tissue is not None else None
    he_tiles = _tiles_from_chops(he_tissue, shape=tile_hw, tuple_index=chop_idx) if he_tissue is not None else None
    num_tiles = len(mx_tiles) if mx_tiles is not None else len(he_tiles)

    instance_processor = segmentation_model.instance_processor
    n_instance_channels = int(segmentation_model.dim_out)

    instance_label_tiles = []
    semantic_logit_tiles = []

    with torch.no_grad():
        for i in tqdm(range(0, num_tiles, bs)):
            batch_len = min(bs, num_tiles - i)
            mx_batch = mx_tiles[i : i + bs] if mx_tiles is not None else [None] * batch_len
            he_batch = he_tiles[i : i + bs] if he_tiles is not None else [None] * batch_len
            channels_batch = [channel_ids] * batch_len

            out = segmentation_model(mx_batch, he_batch, channels_batch)
            logits = torch.cat([out["instance_branch"], out["phenotype_branch"]], dim=1).detach()
            if logits.shape[-2:] != tile_hw:
                logits = F.interpolate(logits, size=tile_hw, mode="bilinear", align_corners=False)

            for tile_logits in logits:
                instance_label = instance_processor.postprocessing(
                    tile_logits[:n_instance_channels],
                    max_seeds=max_seeds,
                    window_size=window_size,
                    cleanup_fragments=True,
                )
                instance_label_tiles.append(instance_label.cpu())
                semantic_logit_tiles.append(tile_logits[n_instance_channels:].cpu())

    pred_instance, _ = _stitch(
        instance_label_tiles, shape=tile_hw, chop_list=chop_idx, offset=ovlp, final_shape=(1, h, w)
    )
    semantic_logits = _stitch_mean(
        semantic_logit_tiles,
        shape=tile_hw,
        chop_list=chop_idx,
        final_shape=(semantic_logit_tiles[0].shape[0], h, w),
    )
    return pred_instance[0], semantic_logits


class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)
