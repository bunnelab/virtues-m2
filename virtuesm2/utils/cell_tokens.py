import numpy as np
import torch
from typing import List, Tuple, Optional

def _get_uniform_crops(img, stride, crop_size=128):
    crops = []
    h, w = img.shape[-2:]
    indices = sorted({(i, j) for i in range(0, h - crop_size + 1, stride) for j in range(0, w - crop_size + 1, stride)})
    last_i, last_j = h - crop_size, w - crop_size
    indices = sorted(set(indices) | {(last_i, j) for j in range(0, w - crop_size + 1, stride)} | {(i, last_j) for i in range(0, h - crop_size + 1, stride)} | {(last_i, last_j)})
    for i, j in indices:
        crops.append(img[:, i:i + crop_size, j:j + crop_size])
    return crops, indices

def _assign_patch_tokens_to_cells(crop_tokens, crop_mask, patch_size):
    cell_tokens = {} 
    weights = {}  
    for i in range(crop_tokens.shape[0]): 
        for j in range(crop_tokens.shape[1]):
            patch_mask = crop_mask[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            unique, counts = np.unique(patch_mask, return_counts=True)
            patch_cell_coverage = dict(zip(unique, counts))
            for cell_id, overlap_pixels in patch_cell_coverage.items():
                if cell_id == 0:  
                    continue
                if cell_id not in weights:
                    weights[cell_id] = []
                weights[cell_id].append(overlap_pixels)
                if cell_id not in cell_tokens:
                    cell_tokens[cell_id] = []
                cell_tokens[cell_id].append(crop_tokens[i, j, :])
    return cell_tokens, weights

def make_protein_tensor(esm_embeddings: dict, uniprot_ids: List[str]) -> torch.Tensor:
    """Build a channel-id tensor from the notebook's ESM lookup dict and uniprot ids.

    Returns shape (num_markers,).
    """
    channel_ids = []
    for uid in uniprot_ids:
        channel_id = esm_embeddings.get(uid)
        if channel_id is None:
            raise KeyError(f"Uniprot id {uid} not found in esm_embeddings")
        channel_ids.append(int(channel_id))
    return torch.tensor(channel_ids, dtype=torch.long)


def _get_patch_tokens_from_model(
    model,
    mx_batch,
    he_batch,
    protein_batch,
    modality: str,
    return_all_layers: bool = False,
):
    sample_img = next((img for img in mx_batch if img is not None), None)
    if sample_img is None:
        sample_img = next(img for img in he_batch if img is not None)
    patches_per_side = sample_img.shape[-1] // model.patch_size
    num_patches = patches_per_side * (sample_img.shape[-2] // model.patch_size)
    tokenized = model.prepare_tokens_with_masks(mx_batch, he_batch, protein_batch)
    outputs = model.forward_features(mx_batch, he_batch, protein_batch, return_all_layers=return_all_layers)
    x_all_layers = outputs["x_all_layers"] if return_all_layers else None
    if x_all_layers is not None:
        for i in range(len(x_all_layers)):
            x_all_layers[i] = torch.cat(
                (
                    x_all_layers[i][:, 1 + model.num_register_tokens : 1 + model.num_register_tokens + num_patches],
                    x_all_layers[i][:, 1 + model.num_register_tokens + num_patches :],
                ),
                dim=-1,
            )

    start = model.num_register_tokens + 1
    tokens_mx = tokenized[:, start : start + num_patches].reshape(tokenized.shape[0], patches_per_side, patches_per_side, -1)
    tokens_he = tokenized[:, start + num_patches :].reshape(tokenized.shape[0], patches_per_side, patches_per_side, -1)
    if modality == "mx":
        return tokens_mx, x_all_layers
    if modality == "he":
        return tokens_he, x_all_layers
    return torch.cat((tokens_mx, tokens_he), dim=-1), x_all_layers


def compute_cell_tokens(
    model,
    *args,
    device: str = "cuda",
    crop_size: int = 224,
    patch_size: int = 14,
    stride: int = 42,
    chunk_size: int = 32,
    modality: str = "both",
    **kwargs,
) -> Tuple[List[int], torch.Tensor, List[torch.Tensor], List[Tuple[int, int]]]:
    '''
    Compute cell tokens using a `virtuesm2` model (VirTuesM2 / VirTuesM2_HF).

    Args:
        model: virtuesm2 model implementing `forward_features` as in `virtuesm2.models.virtuesm2`
        Supports both call styles:
        - compute_cell_tokens(model, mx_img, he_img, channel, segmentation_mask, ...)
        - compute_cell_tokens(model, channel, segmentation_mask, mx_img=..., he_img=..., ...)
        channel: protein embedding tensor of shape (num_markers, D) or (1, num_markers, D)
        segmentation_mask: cell segmentation mask of shape (H, W)
        mx_img: multiplex image tensor of shape (C, H, W)
        he_img: H&E image tensor of shape (3, H, W)
        device: device to run the model on
        crop_size: size of the crops to extract
        patch_size: size of the patches in the model
        stride: stride for extracting crops
        chunk_size: number of crops to process in a chunk
    Returns:
        cell_ids: list of cell ids
        cell_tokens: tensor of shape (num_cells, token_dim)
        crop_tokens: list of tensors of patch summary tokens for each crop
        indices: list of (row, col) indices for each crop
    '''
    assert modality in ("mx", "he", "both"), "modality must be one of 'mx', 'he', 'both'"

    mx_img = kwargs.pop("mx_img", None)
    he_img = kwargs.pop("he_img", None)
    channel = kwargs.pop("channel", None)
    segmentation_mask = kwargs.pop("segmentation_mask", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    if len(args) == 4 and channel is None and segmentation_mask is None and mx_img is None and he_img is None:
        mx_img, he_img, channel, segmentation_mask = args
    elif len(args) == 2 and channel is None and segmentation_mask is None:
        channel, segmentation_mask = args
    elif len(args) != 0:
        raise TypeError(
            "compute_cell_tokens expects either (model, mx_img, he_img, channel, segmentation_mask, ...) or "
            "(model, channel, segmentation_mask, mx_img=..., he_img=..., ...)"
        )

    if channel is None or segmentation_mask is None:
        raise ValueError("channel and segmentation_mask must be provided")

    if modality == "mx":
        if mx_img is None:
            raise ValueError("mx_img must be provided when modality is 'mx'")
        active_img = mx_img
    elif modality == "he":
        if he_img is None:
            raise ValueError("he_img must be provided when modality is 'he'")
        active_img = he_img
    else:
        if mx_img is None or he_img is None:
            raise ValueError("mx_img and he_img must both be provided when modality is 'both'")
        active_img = mx_img

    crops, indices = _get_uniform_crops(active_img, stride, crop_size=crop_size)
    crops = [c.to(device) for c in crops]
    he_crops = None
    if he_img is not None:
        he_crops, he_indices = _get_uniform_crops(he_img, stride, crop_size=crop_size)
        if he_indices != indices:
            raise ValueError("HE image crops do not align with multiplex image crops")
        he_crops = [c.to(device) for c in he_crops]

    channel = channel.to(device).squeeze()
    if channel.ndim != 1:
        raise ValueError("channel must be a 1D tensor of channel ids")

    crop_tokens = []
    for i in range(0, len(crops), chunk_size):
        mx_batch = crops[i : i + chunk_size] if modality != "he" else [None] * len(crops[i : i + chunk_size])
        he_batch = he_crops[i : i + chunk_size] if he_crops is not None else [None] * len(crops[i : i + chunk_size])
        protein_batch = channel.unsqueeze(0).repeat(len(crops[i : i + chunk_size]), 1)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                crop_tokens_chunk, _ = _get_patch_tokens_from_model(
                    model,
                    mx_batch,
                    he_batch,
                    protein_batch,
                    modality=modality,
                )
        crop_tokens.extend([c.cpu() for c in crop_tokens_chunk])

    cell_tokens = {}
    weights = {}

    for crop_token, (row, col) in zip(crop_tokens, indices):
        crop_mask = segmentation_mask[row:row + crop_size, col:col + crop_size]
        crop_cell_tokens, crop_weights = _assign_patch_tokens_to_cells(crop_token, crop_mask, patch_size)
        for cell_id, tokens in crop_cell_tokens.items():
            if cell_id not in cell_tokens:
                cell_tokens[cell_id] = []
            if cell_id not in weights:
                weights[cell_id] = []
            cell_tokens[cell_id].extend(tokens)
            weights[cell_id].extend(crop_weights[cell_id])
    
    avg_cell_tokens = {}
    for cell_id, tokens in cell_tokens.items():
        
        tokens = torch.stack(tokens)
        weights_array = torch.tensor(weights[cell_id], dtype=tokens.dtype, device=tokens.device)
        weights_array = weights_array / weights_array.sum()
        avg_cell_token = torch.sum(tokens * weights_array[:, None], dim=0)
        avg_cell_tokens[cell_id] = avg_cell_token
    cell_ids = list(avg_cell_tokens.keys())
    cell_ids.sort()
    cell_tokens = torch.stack([avg_cell_tokens[cell_id] for cell_id in cell_ids])
    return cell_ids, cell_tokens, crop_tokens, indices