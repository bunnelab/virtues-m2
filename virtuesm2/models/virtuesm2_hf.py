from functools import partial

import torch
from huggingface_hub import PyTorchModelHubMixin

from virtuesm2.models.layers import NestedTensorBlock as Block
from virtuesm2.models.layers.attention import MemEffAttention

from ..utils.utils import load_marker_embeddings
from .multiplex_tokenizer import MultiplexTokenizer
from .virtuesm2 import VirTuesM2


class VirTuesM2_HF(VirTuesM2, PyTorchModelHubMixin):
    """
    HuggingFace compatible version of VirTuesM2. This allows to directly use "from_pretrained" but expects the marker embedding directory to be prepared before.
    """

    def __init__(
        self,
        marker_embedding_dir: str,
        patch_size=14,
        tok_model_dim=1024,
        tok_feedforward_dim=2048,
        tok_encoder_pattern="vv",
        tok_num_encoder_heads=8,
        tok_dropout=0,
        global_crops_size=224,
        layerscale=1.0e-05,
        arch="vit_large",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        num_register_tokens=4,
        he_embed_layer=None,
        interpolate_offset=0.1,
        interpolate_antialias=False,
        ffn_layer="swiglu",
        norm_after_he_tokenizer=True,
    ):
        esm_embds = load_marker_embeddings(marker_embedding_dir)
        esm_embds = esm_embds.to(dtype=torch.float16)
        multiplex_tokenizer = MultiplexTokenizer(
            protein_emb=esm_embds,
            patch_size=patch_size,
            model_dim=tok_model_dim,
            feedforward_dim=tok_feedforward_dim,
            encoder_pattern=tok_encoder_pattern,
            num_encoder_heads=tok_num_encoder_heads,
            dropout=tok_dropout,
            group_layers=True,
        )
        if arch == "vit_large":
            vit_kwargs = dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention))
        else:
            raise NotImplementedError(f"Architecture {arch} not implemented in HF version of ViT")

        vit_kwargs.update(
            img_size=global_crops_size,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            num_register_tokens=num_register_tokens,
            init_values=layerscale,
            block_chunks=block_chunks,
            interpolate_offset=interpolate_offset,
            interpolate_antialias=interpolate_antialias,
            mx_embed_layer=multiplex_tokenizer,
            he_embed_layer=he_embed_layer,
            ffn_layer=ffn_layer,
            norm_after_he_tokenizer=norm_after_he_tokenizer,
        )
        super().__init__(**vit_kwargs)
