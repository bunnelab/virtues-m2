import torch
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
import functools
from collections import OrderedDict


class CacheDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, *args, cache_len: int = 10, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val


SA_BIAS_CACHE = CacheDict(cache_len=20)


def cached_build_block_diagonal_mask(cache_key: str, q_seq_len, kv_seqlen=None):
    """
    q_seqlen: list of sequence lengths for queries
    kv_seqlen: list of sequence lengths for keys and values
    """
    if cache_key in SA_BIAS_CACHE:
        return SA_BIAS_CACHE[cache_key]

    if kv_seqlen is None:
        kv_seqlen = q_seq_len
    attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=q_seq_len, kv_seqlen=kv_seqlen)
    SA_BIAS_CACHE[cache_key] = attn_bias
    return attn_bias


def cached_build_selfattention_bias(cache_key: str, split_mask, use_true_as_query=True):
    """
    split_mask: ... x S tensor of bools
    If cross_attention, False will be used as queries and True as keys and values.
    """
    if cache_key in SA_BIAS_CACHE:
        return SA_BIAS_CACHE[cache_key]

    seq_lens = calc_seq_lens_sums(split_mask, use_true_as_query=use_true_as_query)
    seq_lens = seq_lens.flatten()
    seq_lens = seq_lens[seq_lens != 0]  # Filter-out zero length sequences for self-attention
    seq_lens = seq_lens.tolist()
    attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seq_lens)
    SA_BIAS_CACHE[cache_key] = attn_bias
    return attn_bias


def build_selfattention_bias(split_mask, use_true_as_query=True):
    """
    split_mask: ... x S tensor of bools
    If cross_attention, False will be used as queries and True as keys and values.
    """
    seq_lens = calc_seq_lens_sums(split_mask, use_true_as_query=use_true_as_query)
    seq_lens = seq_lens.flatten()
    seq_lens = seq_lens[seq_lens != 0]  # Filter-out zero length sequences for self-attention
    seq_lens = seq_lens.tolist()
    attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seq_lens)
    return attn_bias


def build_selfattention_bias_channel_concat(split_mask, tokens_per_sequence, use_true_as_query=True):
    """
    This builds a BlockDiagonalMask when in the final dimension of the input tensor, the channels
    of different samples are concatenated and should not attend to each other.
    split_mask: ... x S tensor of bools
    tokens_per_sequence: number of channels per sample summing up to S i.e. S_i
    """
    split_masks = split_mask.split(tokens_per_sequence, dim=-1)  # list of ... x S_i tensors
    seq_lens = [calc_seq_lens_sums(m, use_true_as_query=use_true_as_query) for m in split_masks]

    seq_lens = torch.stack(seq_lens, dim=-1)
    seq_lens = seq_lens[seq_lens != 0]  # Filter-out zero length sequences for self-attention
    seq_lens = seq_lens.tolist()
    attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seq_lens)
    return attn_bias


def build_crossattention_bias(split_mask, use_true_as_query=True):
    if use_true_as_query:
        q_seq_lens = split_mask.sum(-1)
        kv_seq_lens = (~split_mask).sum(-1)
    else:
        q_seq_lens = (~split_mask).sum(-1)
        kv_seq_lens = split_mask.sum(-1)
    q_seq_lens = q_seq_lens.flatten().tolist()
    kv_seq_lens = kv_seq_lens.flatten().tolist()
    if 0 in q_seq_lens or 0 in kv_seq_lens:
        raise ValueError("Cross attention requires at least one 'False' element per sequence")
    attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=q_seq_lens, kv_seqlen=kv_seq_lens)
    return attn_bias


def calc_seq_lens_sums(split_mask, use_true_as_query=True) -> torch.Tensor:
    """
    split_mask: ... x S tensor of bools
    If cross_attention, False will be used as queries and True as keys and values.
    """
    if use_true_as_query:
        seq_lens = split_mask.sum(-1)
    else:
        seq_lens = (~split_mask).sum(-1)
    return seq_lens


def cached_get_non_zero_indices(cache_key: str, x: torch.Tensor):
    """
    x: ... x S tensor of bools
    Returns a list of indices of non-zero elements in the last dimension
    """
    if cache_key in SA_BIAS_CACHE:
        return SA_BIAS_CACHE[cache_key]

    indices = get_non_zero_indices(x)
    SA_BIAS_CACHE[cache_key] = indices
    return indices


def get_non_zero_indices(x: torch.Tensor):
    """
    x: ... x S tensor of bools
    Returns a list of indices of non-zero elements in the last dimension
    """
    return x.nonzero(as_tuple=True)
