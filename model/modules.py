from typing import Union, Optional
import torch
from torch import nn, Tensor

from .layers import PositionalEncoding, Attention
from .blocks import TransBlock


class Coder(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        in_dim: int,
        att_hid_dim: int,
        ffn_hid_dim: int,
        att_n_heads: int = 1,
        att_in_dim_k: Optional[int] = None,
        att_out_dim_v: Optional[int] = None,
        norm_first: bool = True,
    ):
        """n_blocks TransBlocks"""
        super(Coder, self).__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList(
            [
                TransBlock(
                    in_dim=in_dim,
                    att_hid_dim=att_hid_dim,
                    ffn_hid_dim=ffn_hid_dim,
                    att_n_heads=att_n_heads,
                    att_in_dim_k=att_in_dim_k,
                    att_out_dim_v=att_out_dim_v,
                    norm_first=norm_first,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        X_self: Tensor,
        mask_self_bef: Optional[Tensor] = None,
        mask_fill_self_bef: float = -torch.inf,
        mask_self_aft: Optional[Tensor] = None,
        mask_fill_self_aft: float = 0,
        need_self_attn: list[int] = [],
        X_cros: Optional[Tensor] = None,
        mask_cros_bef: Optional[Tensor] = None,
        mask_fill_cros_bef: float = -torch.inf,
        mask_cros_aft: Optional[Tensor] = None,
        mask_fill_cros_aft: float = 0,
        need_cros_attn: list[int] = [],
        need_block_out: list[int] = [],
        dropout: float = 0.0,
    ) -> tuple[Tensor, dict[int, Tensor], dict[int, Tensor], dict[int, Tensor]]:
        """
        Args similar to TransBlock:
        - need_X_attn: indexes(0-based) of blocks of which attention is needed
        - need_block_out: indexes(0-based) of blocks of which output is needed
        ---
        Returns:
        - X: [..., seq_len_q, in_dim_q]
        - dict of self_attn: [..., n_heads, seq_len_q, seq_len_q]
        - dict of cros_attn: [..., n_heads, seq_len_q, seq_len_k]
        - dict of block_out: [..., seq_len_q, in_dim_q]
        """
        self_attn_dict: dict[int, Tensor] = {}
        cros_attn_dict: dict[int, Tensor] = {}
        block_out_dict: dict[int, Tensor] = {}
        for i, block in enumerate(self.blocks):
            X_self, self_attn, cros_attn = block(
                X_self,
                mask_self_bef,
                mask_fill_self_bef,
                mask_self_aft,
                mask_fill_self_aft,
                i in need_self_attn,
                X_cros,
                mask_cros_bef,
                mask_fill_cros_bef,
                mask_cros_aft,
                mask_fill_cros_aft,
                i in need_cros_attn,
                dropout,
            )
            if i in need_self_attn:
                self_attn_dict[i] = self_attn
            if i in need_cros_attn:
                cros_attn_dict[i] = cros_attn
            if i in need_block_out:
                block_out_dict[i] = X_self
        return X_self, self_attn_dict, cros_attn_dict, block_out_dict


class TokenPrenet(nn.Module):
    def __init__(
        self,
        n_toks: int,
        conv_n_layers: int,
        conv_k_size: int,
        out_dim: int,
        embed_dim: Optional[int] = None,
        pad_idx: Optional[int] = None,
        conv_hid_dim: Optional[int] = None,
    ):
        """
        Embedding -> [Conv1d -> BatchNorm -> ReLU] * conv_n_layers -> Linear
        - conv_k_size should be odd to keep padding symmetric
        - if embed_dim is None, default = out_dim
        - pad_idx is index of padding token in embedding
        - if conv_hid_dim is None, default = out_dim
        """
        super(TokenPrenet, self).__init__()
        assert conv_k_size % 2 == 1, f"conv_k_size({conv_k_size}) should be odd"
        if embed_dim is None:
            embed_dim = out_dim
        if conv_hid_dim is None:
            conv_hid_dim = out_dim
        self.embedding = nn.Embedding(n_toks, embed_dim, pad_idx)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim if i == 0 else conv_hid_dim,
                        conv_hid_dim,
                        conv_k_size,
                        padding=(conv_k_size - 1) // 2,
                    ),
                    nn.BatchNorm1d(conv_hid_dim),
                )
                for i in range(conv_n_layers)
            ]
        )
        self.linear = nn.Linear(conv_hid_dim, out_dim)

    def forward(self, X: Tensor, dropout: float = 0.0) -> Tensor:
        # [batch_size, seq_len, embed_dim]
        X = self.embedding(X)
        # [batch_size, embed_dim, seq_len]
        X = X.transpose(-2, -1)
        for conv in self.convs:
            # [batch_size, conv_hid_dim, seq_len]
            X = torch.dropout(torch.relu(conv(X)), dropout, self.training)
        # [batch_size, seq_len, out_dim]
        X = self.linear(X.transpose(-2, -1))
        return X


class LinearNet(nn.Module):
    def __init__(self, n_layers: int, in_dim: int, hid_dim: int, out_dim: int):
        """
        n_layers Linear with relu: in_dim -> hid_dim -> ... -> hid_dim -> out_dim
        """
        super(LinearNet, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    in_dim if i == 0 else hid_dim,
                    out_dim if i == n_layers - 1 else hid_dim,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, X: Tensor, dropout: float = 0.0) -> Tensor:
        """map X from [..., in_dim] to [..., out_dim]"""
        # [..., in_dim]
        for layer in self.layers:
            # [..., hid_dim]
            X = torch.dropout(torch.relu(layer(X)), dropout, self.training)
        # [..., out_dim]
        return X


class MelPostnet(nn.Module):
    def __init__(
        self, in_dim: int, conv_n_layers: int, conv_k_size: int, conv_hid_dim: int
    ):
        super(MelPostnet, self).__init__()
        assert conv_k_size % 2 == 1, f"conv_k_size({conv_k_size}) should be odd"
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_dim if i == 0 else conv_hid_dim,
                        in_dim if i == conv_n_layers - 1 else conv_hid_dim,
                        conv_k_size,
                        padding=(conv_k_size - 1) // 2,
                    ),
                    nn.BatchNorm1d(in_dim if i == conv_n_layers - 1 else conv_hid_dim),
                )
                for i in range(conv_n_layers)
            ]
        )

    def forward(
        self,
        X: Tensor,
        mask: Optional[Tensor] = None,
        mask_fill: float = 0.0,
        dropout: float = 0.0,
    ) -> Tensor:
        """
        - mask: [batch_size, conv_hid_dim, seq_len] as bool
        """
        # [batch_size, in_dim, seq_len]
        X = X.transpose(-2, -1)
        for i, conv in enumerate(self.convs):
            # [batch_size, conv_hid_dim, seq_len]
            X = conv(X)
            if i != len(self.convs) - 1:
                X = torch.tanh(X)
            X = torch.dropout(X, dropout, self.training)
            if mask is not None:
                X = X.masked_fill(mask, mask_fill)
        # [batch_size, seq_len, in_dim]
        X = X.transpose(-2, -1)
        return X
