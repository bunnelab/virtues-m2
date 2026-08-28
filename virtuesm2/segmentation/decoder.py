from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from instanseg.utils.loss.instanseg_loss import InstanSeg
from instanseg.utils.tiling import _chops, _stitch_mean, _tiles_from_chops
from tqdm import tqdm

from .utils import Conv2DBlock, Deconv2DBlock, segment_large_tissue


def build_segmentation_model(vm2_model: nn.Module, num_celltypes: int, modality: str, ckpt_path: str = None, mlp_width: int = 64) -> nn.Module:
    assert modality in {"multiplex", "he", "multimodal"}, f"Invalid modality: {modality}. Must be one of 'multiplex', 'he', or 'multimodal'."
    instanseg = InstanSeg(
        n_sigma=2,
        window_size=128,
    )
    segmentation_model = VM2_Segmentation(
        virtuesm2_model=vm2_model,
        dim_out=instanseg.dim_out,
        num_celltypes=num_celltypes,
        extract_layers=[6, 12, 18, 24],
        mode=modality,
        model_dim=2048 if modality == "multimodal" else 1024,
    ).cpu()
    segmentation_model = instanseg.initialize_pixel_classifier(segmentation_model, MLP_width=mlp_width)
    segmentation_model = segmentation_model.to("cuda")
    
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        segmentation_model.load_state_dict(state_dict, strict=True)
    segmentation_model.eval()
    
    segmentation_model.instance_processor = instanseg
    return segmentation_model


class SegmentationDecoder(nn.Module):
    """CellViT-based cell segmentation model.

    For VirTues-M2, the tokens are concatenated along the feature dimension to incorporate both modalities.
    Skip connections are shared between branches, but each network has a distinct encoder.

    Args:
        num_nuclei_classes (int): Number of nuclei classes (including background)
        embed_dim (int): Embedding dimension of backbone ViT
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int
    ):
        # For simplicity, we will assume that extract layers must have a length of 4
        super().__init__()

        self.patch_size = 14
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12),
            Deconv2DBlock(self.skip_dim_12, 128),
            Deconv2DBlock(128, 64),
        )  # skip connection 1

        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12),
            Deconv2DBlock(self.skip_dim_12, 128),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11),
            Deconv2DBlock(self.skip_dim_11, 256),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(Deconv2DBlock(self.embed_dim, self.bottleneck_dim))  # skip connection 3

        self.decoder = self.create_upsampling_branch(out_channels)

    def forward(self, x_patches, extracted_layers) -> dict:
        """Forward pass
        """

        z0 = x_patches
        z1 = extracted_layers[0]
        z2 = extracted_layers[1]
        z3 = extracted_layers[2]
        z4 = extracted_layers[3]

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        patch_dim = [int(d ** 0.5) for d in [z0.shape[-2], z0.shape[-2]]]
        z4 = z4.transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2.transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1.transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z0 = z0.transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        return self._forward_upsample(z0, z1, z2, z3, z4, self.decoder)

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256),
            Conv2DBlock(256, 256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128),
            Conv2DBlock(128, 128),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64),
            Conv2DBlock(64, 64),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder
    
class VM2_Segmentation(nn.Module):
    def __init__(self, virtuesm2_model, dim_out: int, num_celltypes=None, extract_layers=[4, 8, 12, 16], model_dim=1024, mode="multiplex"):
        super().__init__()
        assert len(extract_layers) == 4, "Provide 4 layers for skip connections"

        self.virtuesm2_model = virtuesm2_model
        self.dim_out = dim_out
        self.decoder = SegmentationDecoder(embed_dim=model_dim, out_channels=dim_out)
        self.extract_layers = torch.tensor(extract_layers)
        self.mode = mode

        self.num_celltypes = num_celltypes
        if num_celltypes is not None:
            self.decoder_phenotypes = SegmentationDecoder(embed_dim=model_dim, out_channels=num_celltypes)

    def _encode(self, mx_images, he_images, channels, mode="multiplex"):
        mx_image = mx_images[0]
        he_image = he_images[0]
        num_patches = mx_image.shape[1] * mx_image.shape[2] // (14*14) if mx_image is not None else he_image.shape[1] * he_image.shape[2] // (14*14)
        if mode == "multiplex":
            he_images = [None] * len(mx_images)
        elif mode == "he":
            mx_images = [None] * len(he_images)
    
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                mx_images = [mx_img.cuda(non_blocking=True) if mx_img is not None else None for mx_img in mx_images]
                he_images = [he_img.cuda(non_blocking=True) if he_img is not None else None for he_img in he_images]
                # Get the tokenized multiplex and H&E inputs
                tokenized_img = self.virtuesm2_model.prepare_tokens_with_masks(mx_images, he_images, channels)
                tokens_mx = tokenized_img[:, self.virtuesm2_model.num_register_tokens + 1: + self.virtuesm2_model.num_register_tokens + 1 + num_patches]
                tokens_he = tokenized_img[:, self.virtuesm2_model.num_register_tokens + 1 + num_patches:]

                # Apply VirTues-M2
                mm_embeddings = self.virtuesm2_model.forward_features(mx_images, he_images, channels, return_all_layers=True)
                x_all_layers = mm_embeddings["x_all_layers"]
                
                # Decoder images into segmentation maps
                # patches_concat = torch.cat((tokens_mx, tokens_he), dim=-1)
                if mode == "multiplex":
                    latent_patches = tokens_mx
                    for i in range(len(x_all_layers)):
                        x_all_layers[i] = x_all_layers[i][:, 1 + self.virtuesm2_model.num_register_tokens : 1 + self.virtuesm2_model.num_register_tokens + num_patches]
                elif mode == "he":
                    latent_patches = tokens_he
                    for i in range(len(x_all_layers)):
                        x_all_layers[i] = x_all_layers[i][:, 1 + self.virtuesm2_model.num_register_tokens + num_patches:]
                else:
                    latent_patches = torch.cat((tokens_mx, tokens_he), dim=-1)
                    for i in range(len(x_all_layers)):
                        x_all_layers[i] = torch.cat((x_all_layers[i][:, 1 + self.virtuesm2_model.num_register_tokens : 1 + self.virtuesm2_model.num_register_tokens + num_patches],
                                                    x_all_layers[i][:, 1 + self.virtuesm2_model.num_register_tokens + num_patches:]),
                                                    dim=-1)
                x_all_layers = torch.stack(x_all_layers)
        return latent_patches, x_all_layers[self.extract_layers - 1]

    def forward(self, mx_images, he_images, channels):
        out_dict = {}
        z0, extracted_layers = self._encode(mx_images, he_images, channels, self.mode)
        out_dict["instance_branch"] = self.decoder(z0, extracted_layers)
        if self.num_celltypes is not None:
            phenotypes = self.decoder_phenotypes(z0, extracted_layers)
            out_dict["phenotype_branch"] = phenotypes

        return out_dict

    @torch.no_grad()
    def segment_tile(self, mx_image, he_image, channel_ids):
        """
        Computes cell instance segmentation and phenotype logits for a single tile.

        Args:
            mx_image (torch.Tensor | None): Multiplexed image tile of shape (C,H,W), or None if the
                decoder's mode does not use the multiplex branch.
            he_image (torch.Tensor | None): H&E image tile of shape (3,H,W), or None if the decoder's
                mode does not use the H&E branch.
            channel_ids (torch.Tensor | None): Channel ids corresponding to the channels in mx_image.
                The tensor should be of shape (C,).
        Returns:
            pred_instance (torch.Tensor): The predicted instance segmentation mask. The tensor will be of
                shape (H,W) with integer values representing different instances.
            semantic_logits (torch.Tensor): The predicted phenotype segmentation logits. The tensor will
                be of shape (num_celltypes,H,W).
        """
        assert self.num_celltypes is not None, "segment_tile requires a phenotype decoder (num_celltypes must be set)"
        out = self.forward([mx_image], [he_image], [channel_ids])
        inst_logits = out["instance_branch"][0]
        pred_instance = self.instance_processor.postprocessing(inst_logits, window_size=64, cleanup_fragments=True)[0]
        semantic_logits = out["phenotype_branch"][0]
        return pred_instance, semantic_logits

    @torch.no_grad()
    def segment_tissue(
        self,
        mx_tissue,
        he_tissue,
        channel_ids,
        tile_size: int,
        overlap: int,
        batch_size: int,
        large_tissue_threshold: int = 1500,
    ):
        """
        Computes cell instance segmentation and phenotype logits for a large tissue image by processing
        it in tiles.

        If the tissue's longer side exceeds `large_tissue_threshold` pixels, delegates to
        `segment_large_tissue` instead, which segments and stitches instances tile by tile, at the cost
        of needing to match instance identities across tile borders.

        Args:
            mx_tissue (torch.Tensor | None): A large multiplexed tissue image of shape (C,H,W), or None
                if the decoder's mode does not use the multiplex branch.
            he_tissue (torch.Tensor | None): A large H&E tissue image of shape (3,H,W), or None if the
                decoder's mode does not use the H&E branch.
            channel_ids (torch.Tensor | None): Channel ids corresponding to the channels in mx_tissue.
                The tensor should be of shape (C,).
            tile_size (int): The width/height of the tiles the image is split into.
            overlap (int): The overlap (in pixels) between neighbouring tiles.
            batch_size (int): The number of tiles processed per batch.
            large_tissue_threshold (int, optional): If the tissue's longer side exceeds this threshold,
                delegates to `segment_large_tissue`. Defaults to 1500.

        Returns:
            pred_instance (torch.Tensor): The predicted instance segmentation mask for the entire tissue.
                The tensor will be of shape (H,W) with integer values representing different instances.
            semantic_logits (torch.Tensor): The predicted phenotype segmentation logits for the entire
                tissue. The tensor will be of shape (num_celltypes,H,W).
        """
        assert self.num_celltypes is not None, "segment_tissue requires a phenotype decoder (num_celltypes must be set)"
        reference_tissue = mx_tissue if mx_tissue is not None else he_tissue
        h, w = int(reference_tissue.shape[-2]), int(reference_tissue.shape[-1])

        if max(h, w) > large_tissue_threshold:
            pred_instance, semantic_logits = segment_large_tissue(
                mx_tissue,
                he_tissue,
                self,
                channel_ids,
                tile=tile_size,
                ovlp=overlap,
                bs=batch_size,
            )
            return pred_instance.cpu(), semantic_logits.cpu()

        tile_hw = (min(tile_size, h), min(tile_size, w))
        chop_idx = _chops(reference_tissue.shape, shape=tile_hw, overlap=2 * overlap)
        mx_tiles = _tiles_from_chops(mx_tissue, shape=tile_hw, tuple_index=chop_idx) if mx_tissue is not None else None
        he_tiles = _tiles_from_chops(he_tissue, shape=tile_hw, tuple_index=chop_idx) if he_tissue is not None else None
        num_tiles = len(mx_tiles) if mx_tiles is not None else len(he_tiles)

        logits_tiles = []

        for i in tqdm(range(0, num_tiles, batch_size)):
            batch_len = min(batch_size, num_tiles - i)
            mx_batch = mx_tiles[i : i + batch_size] if mx_tiles is not None else [None] * batch_len
            he_batch = he_tiles[i : i + batch_size] if he_tiles is not None else [None] * batch_len
            channels_batch = [channel_ids] * batch_len

            out = self.forward(mx_batch, he_batch, channels_batch)
            pred = torch.cat([out["instance_branch"], out["phenotype_branch"]], dim=1).detach()
            if pred.shape[-2:] != tile_hw:
                pred = F.interpolate(pred, size=tile_hw, mode="bilinear", align_corners=False)
            logits_tiles.extend([p for p in pred])

        stitched_logits = _stitch_mean(
            logits_tiles,
            shape=tile_hw,
            chop_list=chop_idx,
            final_shape=(logits_tiles[0].shape[0], h, w),
        )

        inst_logits = stitched_logits[: self.dim_out]
        pred_instance = self.instance_processor.postprocessing(inst_logits, window_size=64, cleanup_fragments=True, max_seeds=30000)[0]
        semantic_logits = stitched_logits[self.dim_out :, :, :]
        return pred_instance.cpu(), semantic_logits.cpu()