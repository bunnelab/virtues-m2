from scipy.ndimage import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt
import numpy as np
import torch
from torch import nn


def create_instance_map(binary_mask, vertical_dist, horizontal_dist, min_distance=5, binary_threshold=0.5, distance_threshold=0.3):
    """
    Create instance segmentation from binary mask and horizontal/vertical distance maps using watershed.
    Following CellViT/HoVer-Net approach with HV maps.

    Args:
        binary_mask: (H, W) Binary segmentation probability map
        vertical_dist: (H, W) Vertical distance map
        horizontal_dist: (H, W) Horizontal distance map
        min_distance: Minimum distance between detected peaks (cell centers)
        binary_threshold: Threshold for binary mask
        distance_threshold: Threshold for distance gradient magnitude to find seeds

    Returns:
        instance_map: (H, W) Instance segmentation with each cell having a unique ID
    """
    # Threshold binary mask
    binary_filled = (binary_mask > binary_threshold).astype(np.uint8)

    # Combine horizontal and vertical distance maps to get gradient magnitude
    # This represents the energy landscape where cells have high values at centers
    gradient_magnitude = np.sqrt(vertical_dist**2 + horizontal_dist**2)

    # Compute distance transform from binary mask as fallback
    dist_transform = distance_transform_edt(binary_filled)

    # Use combination of gradient magnitude and distance transform
    # Distance transform gives us distance from edges, high at cell centers
    combined_distance = dist_transform * (1 + gradient_magnitude)

    # Detect peaks (cell centers)
    coordinates = peak_local_max(
        combined_distance,
        min_distance=min_distance,
        threshold_abs=distance_threshold,
        exclude_border=False,
        labels=binary_filled,
    )
    # Create markers for watershed
    markers = np.zeros_like(binary_filled, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[coord[0], coord[1]] = idx + 1

    # If no seeds found, try basic connected components
    if markers.max() == 0:
        labeled, num_features = label(binary_filled)
        return labeled.astype(np.int32)

    # Apply watershed algorithm
    # Use negative combined distance so watershed treats high values as basins
    instance_map = watershed(-combined_distance, markers, mask=binary_filled)

    return instance_map.astype(np.int32)


def batch_create_instance_maps(binary_probs, distance_probs, **kwargs):
    """
    Create instance maps for a batch of predictions.

    Args:
        binary_probs: (B, H, W) or (B, 2, H, W) Binary probabilities
        distance_probs: (B, 2, H, W) Distance probabilities with [vertical, horizontal] channels
        **kwargs: Additional arguments for create_instance_map

    Returns:
        instance_maps: (B, H, W) Instance segmentation maps
    """
    # Handle different input shapes for binary
    if len(binary_probs.shape) == 4:
        binary_probs = binary_probs[:, 1]  # Take cell class probability

    # Handle distance probs - expect (B, 2, H, W) with [vertical, horizontal]
    if len(distance_probs.shape) == 4 and distance_probs.shape[1] == 2:
        vertical_probs = distance_probs[:, 0]
        horizontal_probs = distance_probs[:, 1]
    else:
        raise ValueError(f"Expected distance_probs shape (B, 2, H, W), got {distance_probs.shape}")

    B = binary_probs.shape[0]
    H, W = binary_probs.shape[1:3]

    instance_maps = np.zeros((B, H, W), dtype=np.int32)

    for i in range(B):
        binary = binary_probs[i]
        vertical_dist = vertical_probs[i]
        horizontal_dist = horizontal_probs[i]

        # Convert to numpy if needed
        if isinstance(binary, torch.Tensor):
            binary = binary.cpu().numpy()
        if isinstance(vertical_dist, torch.Tensor):
            vertical_dist = vertical_dist.cpu().numpy()
        if isinstance(horizontal_dist, torch.Tensor):
            horizontal_dist = horizontal_dist.cpu().numpy()

        instance_maps[i] = create_instance_map(binary, vertical_dist, horizontal_dist, **kwargs)

    return instance_maps


def remove_small_regions_batched(batch_labeled_images, min_size):
    """
    Remove all labeled regions in a batch of images with fewer than min_size pixels.

    Parameters:
    - batch_labeled_images: 3D numpy array (B, H, W) where each region has a unique integer label (0 for background)
    - min_size: minimum number of pixels a region must have to be kept

    Returns:
    - filtered_batch: 3D numpy array (B, H, W) with small regions removed (set to 0)
    """
    filtered_batch = np.copy(batch_labeled_images)
    B = batch_labeled_images.shape[0]

    for b in range(B):
        filtered_batch[b] = remove_small_regions(batch_labeled_images[b], min_size)

    return filtered_batch


def remove_small_regions(labeled_image, min_size):
    """
    Remove all labeled regions in an image with fewer than min_size pixels.

    Parameters:
    - labeled_image: 2D numpy array where each region has a unique integer label (0 for background)
    - min_size: minimum number of pixels a region must have to be kept

    Returns:
    - filtered_image: 2D numpy array with small regions removed (set to 0)
    """
    filtered_image = np.copy(labeled_image)
    region_ids, counts = np.unique(labeled_image, return_counts=True)

    # Iterate through each region ID except background (0)
    for region_id, count in zip(region_ids, counts):
        if region_id == 0:  # skip background
            continue
        if count < min_size:
            filtered_image[filtered_image == region_id] = 0

    return filtered_image


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
