import torch.nn as nn
from einops import rearrange

from .attention_xformers import MHAwithPosEmb
from .basics import build_feedforward
from .mask_utils_xformers import cached_build_block_diagonal_mask


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, activation="gelu", bias=True, inbuilt_pos_emb="absolute", num_layers=1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model, num_heads, dim_feedforward, dropout=dropout, activation=activation, bias=bias, inbuilt_pos_emb=inbuilt_pos_emb
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, src_pos=None, mask=None, src_key_padding_mask=None, return_attention=False, before_softmax=False):
        """
        Args:
            src: (B, S, d_model) source sequence
            attn_mask: (B, S, S) boolean mask or float mask
            key_padding_mask: (B, S) boolean mask or float mask
        """
        if return_attention:
            for layer in self.layers:
                src, attention = layer(
                    src,
                    src_pos=src_pos,
                    src_key_padding_mask=src_key_padding_mask,
                    mask=mask,
                    return_attention=return_attention,
                    before_softmax=before_softmax,
                )
            return src, attention

        else:
            for layer in self.layers:
                src = layer(
                    src,
                    src_pos=src_pos,
                    src_key_padding_mask=src_key_padding_mask,
                    mask=mask,
                    return_attention=return_attention,
                    before_softmax=before_softmax,
                )
            return src


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="gelu", bias=True, inbuilt_pos_emb="absolute"):
        super(TransformerEncoderBlock, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.activation = activation

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.multi_head_attention = MHAwithPosEmb(embed_dim=d_model, num_heads=nhead, dropout=dropout, bias=bias, inbuilt_pos_emb=inbuilt_pos_emb)
        self.feedforward = build_feedforward(d_model, dim_feedforward, activation, dropout)

        self.layernorm1 = nn.LayerNorm(d_model, bias=bias)
        self.layernorm2 = nn.LayerNorm(d_model, bias=bias)

    def forward(self, src, src_pos=None, mask=None, src_key_padding_mask=None, return_attention=False, before_softmax=False):
        """
        Args:
            src: (B, S, d_model) source sequence
            attn_mask: (B, S, S) boolean mask or float mask
            key_padding_mask: (B, S) boolean mask or float mask
        """

        """ Post LN Attention 
        src = self.layernorm1(src + self.multi_head_attention(query=src, key=src, value=src, query_pos=src_pos, key_pos=src_pos, mask=mask, key_padding_mask=src_key_padding_mask))
        src = self.layernorm2(src + self.feedforward(src))
        """
        # Pre-LN MHA
        lsrc = self.layernorm1(src)
        if return_attention:
            mha, attention = self.multi_head_attention(
                query=lsrc,
                key=lsrc,
                value=lsrc,
                query_pos=src_pos,
                key_pos=src_pos,
                mask=mask,
                key_padding_mask=src_key_padding_mask,
                return_attention=return_attention,
                before_softmax=before_softmax,
            )
            src = src + mha
        else:
            src = src + self.multi_head_attention(
                query=lsrc, key=lsrc, value=lsrc, query_pos=src_pos, key_pos=src_pos, mask=mask, key_padding_mask=src_key_padding_mask
            )
        lsrc = self.layernorm2(src)
        src = src + self.feedforward(lsrc)

        if return_attention:
            return src, attention
        return src


class MarkerAttentionEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope", num_layers=1):
        super().__init__()
        self.encoder_layer = TransformerEncoder(
            model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb, num_layers=num_layers
        )

    def forward(self, x, pos, return_attention=False, before_softmax=False, **kwargs):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B S) C D")
        pos = rearrange(pos, "B C S D -> (B S) C D")
        if return_attention:
            x, attention = self.encoder_layer(src=x, src_pos=pos, return_attention=return_attention, before_softmax=before_softmax)
            x = rearrange(x, "(B S) C D -> B C S D", B=B)
            return x, attention
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "(B S) C D -> B C S D", B=B)
        return x

    def forward_cc(self, x, pos, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        """
        S = x.shape[1]
        q_seq_lens = channels_per_sample * S
        x = rearrange(x, "C S D -> (S C) D").unsqueeze(0)
        pos = rearrange(pos, "C S D -> (S C) D").unsqueeze(0)
        attn_bias = cached_build_block_diagonal_mask("MarkerAttention_cc", q_seq_len=q_seq_lens)
        x = self.encoder_layer(src=x, src_pos=pos, mask=attn_bias)
        x = x.squeeze(0)
        x = rearrange(x, "(S C) D -> C S D", S=S)
        return x


class FullAttentionEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope", num_layers=1):
        super().__init__()
        self.encoder_layer = TransformerEncoder(
            model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb, num_layers=num_layers
        )

    def forward(self, x, pos, **kwargs):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> B (C S) D")
        pos = rearrange(pos, "B C S D -> B (C S) D")
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "B (C S) D -> B C S D", C=C)
        return x

    def forward_cc(self, x, pos, channels_per_sample, **kwargs):
        """
        x: CxSxD
        pos: CxSx2
        """
        S = x.shape[1]
        q_seq_lens = [c * S for c in channels_per_sample]
        x = rearrange(x, "C S D -> (C S) D").unsqueeze(0)
        pos = rearrange(pos, "C S D -> (C S) D").unsqueeze(0)
        attn_bias = cached_build_block_diagonal_mask("FullAttention_cc", q_seq_len=q_seq_lens)
        x = self.encoder_layer(src=x, src_pos=pos, mask=attn_bias)
        x = x.squeeze(0)
        x = rearrange(x, "(C S) D -> C S D", S=S)

        return x
