from typing import Optional
import torch
from torch import nn, Tensor

from .layers import Attention


class TransBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        att_hid_dim: int,
        ffn_hid_dim: int,
        att_n_heads: int = 1,
        att_in_dim_k: Optional[int] = None,
        att_out_dim_v: Optional[int] = None,
        norm_first: bool = True,
    ):
        """
        Args:
        - in_dim: input dimension of block
        - att_hid_dim: hidden dimension of query and key to map to
        - ffn_hid_dim: hidden dimension of feed forward net
        - att_n_heads: number of heads in Attention
        - att_in_dim_k:
            if not None, a cross-attention is added,
            int for input dimension of k/v in cross-attention
        - att_out_dim_v:
            output dimension of value, default = in_dim
            when att_n_heads==1, it should be equal to in_dim
        - norm_first: whether to apply normalization before sublayer
        """
        super(TransBlock, self).__init__()
        self.in_dim = in_dim
        if att_n_heads == 1 or att_out_dim_v is None:
            att_out_dim_v = in_dim
        self.self_attn_layer = Attention(
            in_dim, att_hid_dim, att_out_dim_v, None, att_n_heads, in_dim
        )
        self.self_attn_norm = nn.LayerNorm(in_dim)
        if att_in_dim_k is not None:
            self.has_cross = True
            self.cros_attn_layer = Attention(
                in_dim, att_hid_dim, att_out_dim_v, att_in_dim_k, att_n_heads, in_dim
            )
            self.cros_attn_norm = nn.LayerNorm(in_dim)
        else:
            self.has_cross = False
        self.post_layer = nn.Sequential(
            nn.Linear(in_dim, ffn_hid_dim), nn.ReLU(), nn.Linear(ffn_hid_dim, in_dim)
        )
        self.post_norm = nn.LayerNorm(in_dim)
        self.norm_first = norm_first

    def forward(
        self,
        X_self: Tensor,
        mask_self_bef: Optional[Tensor] = None,
        mask_fill_self_bef: float = -torch.inf,
        mask_self_aft: Optional[Tensor] = None,
        mask_fill_self_aft: float = 0,
        need_self_attn: bool = False,
        X_cros: Optional[Tensor] = None,
        mask_cros_bef: Optional[Tensor] = None,
        mask_fill_cros_bef: float = -torch.inf,
        mask_cros_aft: Optional[Tensor] = None,
        mask_fill_cros_aft: float = 0,
        need_cros_attn: bool = False,
        dropout: float = 0.0,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Args:
        - X_self: [..., seq_len_q, in_dim_q], input of self-attention layer
        - X_cros: [..., seq_len_k, in_dim_k], input(k/v) of cross-attention layer
        - mask_X_Y: mask for X-attention layer before/after softmax
        - mask_fill_X_Y: fill value for masked position
        - need_X_attn: whether need X-attention
        - dropout: dropout probability for each sublayer when training
        ---
        Returns:
        - X: [..., seq_len_q, in_dim_q]
        - self_attn: [..., n_heads, seq_len_q, seq_len_q] if need_self_attn else None
        - cros_attn: [..., n_heads, seq_len_q, seq_len_k] if need_cros_attn else None
        """
        if self.norm_first:
            X_self: Tensor = self.post_norm(X_self)
        X_out, self_attn = self.self_attn_layer(
            X_self,
            None,
            mask_self_bef,
            mask_fill_self_bef,
            mask_self_aft,
            mask_fill_self_aft,
            need_self_attn,
        )
        X_self = X_self + torch.dropout(X_out, dropout, self.training)
        X_self: Tensor = self.self_attn_norm(X_self)
        if self.has_cross:
            X_out, cros_attn = self.cros_attn_layer(
                X_self,
                X_cros,
                mask_cros_bef,
                mask_fill_cros_bef,
                mask_cros_aft,
                mask_fill_cros_aft,
                need_cros_attn,
            )
            X_self = X_self + torch.dropout(X_out, dropout, self.training)
            X_self = self.cros_attn_norm(X_self)
        else:
            cros_attn = None
        X_self = X_self + torch.dropout(self.post_layer(X_self), dropout, self.training)
        if not self.norm_first:
            X_self = self.post_norm(X_self)
        return X_self, self_attn, cros_attn


class FFTBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        att_hid_dim: int,
        conv_hid_dim: int,
        conv_k_size: int,
        att_n_heads: int = 1,
        att_in_dim_k: Optional[int] = None,
        att_out_dim_v: Optional[int] = None,
        norm_first: bool = True,
    ):
        """
        Args:
        - in_dim: input dimension of block
        - att_hid_dim: hidden dimension of query and key to map to
        - conv_hid_dim: hidden dimension of conv1d postnet
        - conv_k_size: kernel size of conv1d postnet
        - att_n_heads: number of heads in Attention
        - att_out_dim_v:
            output dimension of value, default = in_dim
            when att_n_heads==1, it should be equal to in_dim
        - norm_first: whether to apply normalization before sublayer
        """
        super(FFTBlock, self).__init__()
        assert conv_k_size % 2 == 1, "kern size should be odd"
        self.in_dim = in_dim
        self.is_cross = att_in_dim_k is not None
        if att_n_heads == 1 or att_out_dim_v is None:
            att_out_dim_v = in_dim
        self.attn_layer = Attention(
            in_dim, att_hid_dim, att_out_dim_v, att_in_dim_k, att_n_heads, in_dim
        )
        self.attn_norm = nn.LayerNorm(in_dim)
        self.post_layer = nn.Sequential(
            nn.Conv1d(
                in_dim, conv_hid_dim, conv_k_size, padding=(conv_k_size - 1) // 2
            ),
            nn.ReLU(),
            nn.Conv1d(
                conv_hid_dim, in_dim, conv_k_size, padding=(conv_k_size - 1) // 2
            ),
        )
        self.post_norm = nn.LayerNorm(in_dim)
        self.norm_first = norm_first

    def forward(
        self,
        Xq: Tensor,
        Xk: Optional[Tensor] = None,
        mask_bef: Optional[Tensor] = None,
        mask_fill_bef: float = -torch.inf,
        mask_aft: Optional[Tensor] = None,
        mask_fill_aft: float = 0,
        need_attn: bool = False,
        dropout: float = 0.0,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Args:
        - X: [..., seq_len, in_dim]
        - mask_X: mask for attention layer before/after softmax
        - mask_fill_X: fill value for masked position
        - need_attn: whether need attention
        - dropout: dropout probability for each sublayer when training
        ---
        Returns:
        - X: [..., seq_len, in_dim]
        - attn: [..., n_heads, seq_len, seq_len] if need_attn else None
        """
        if self.norm_first:
            Xq = self.post_norm(Xq)
        X_out, attn = self.attn_layer(
            Xq=Xq,
            Xk=Xk,
            mask_bef=mask_bef,
            mask_fill_bef=mask_fill_bef,
            mask_aft=mask_aft,
            mask_fill_aft=mask_fill_aft,
            need_attention=need_attn,
        )
        # [..., seq_len, in_dim]
        Xq = Xq + torch.dropout(X_out, dropout, self.training)
        Xq = self.attn_norm(Xq)
        Xq = Xq + torch.dropout(
            self.post_layer(Xq.transpose(-2, -1)), dropout, self.training
        ).transpose(-2, -1)
        if not self.norm_first:
            Xq = self.post_norm(Xq)
        return Xq, attn
