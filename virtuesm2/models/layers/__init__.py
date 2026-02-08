from virtuesm2.utils.utils import is_rank0
from loguru import logger

from .dino_head import DINOHead
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from .block import NestedTensorBlock

from .attention import MemEffAttention
from .attention_qkv_separate import MemEffAttention_qkv_separate
from .multiplex_tokenizer.transformer_xformers import MarkerAttentionEncoderBlock, FullAttentionEncoderBlock
from .multiplex_tokenizer.mask_utils_xformers import SA_BIAS_CACHE


try:
    import xformers


    USE_XFORMERS = True
    if is_rank0():
        logger.info(f"Using xformers version {xformers.__version__} for attention layers.")
except ImportError as e:
    logger.error(f"Failed to import xformers: {e}")
    logger.error("xformers not found, there is currently no fallback. xformers needs to be installed.")
