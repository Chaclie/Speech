from typing import Optional, Union
import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, max_in_dim: int):
        """
        Sin/Cos Position Encoding
        - aligned_in_dim = ceil(max_in_dim / 2) * 2
        - PE[pos, 2i]   = sin(pos / (max_seq_len ^ (2i / aligned_in_dim)))
        - PE[pos, 2i+1] = cos(pos / (max_seq_len ^ (2i / aligned_in_dim)))
        """
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_in_dim = max_in_dim
        aligned_in_dim: int = max_in_dim if max_in_dim % 2 == 0 else max_in_dim + 1
        # [max_seq_len, aligned_in_dim]
        pe = torch.arange(max_seq_len).view(max_seq_len, 1).repeat(1, aligned_in_dim)
        pe = pe.to(dtype=torch.float)
        freq = torch.zeros(1, aligned_in_dim)  # [1, aligned_in_dim]
        freq[:, 0::2] = freq[:, 1::2] = torch.arange(0, aligned_in_dim, 2)
        pe /= torch.exp(freq / aligned_in_dim * torch.log(torch.tensor(max_seq_len)))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pe[:, 0::2]), torch.cos(pe[:, 1::2])
        # [max_seq_len, max_in_dim]
        pe = pe[:, :max_in_dim]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, X: Tensor, alpha: Union[float, Tensor] = 1.0) -> Tensor:
        """
        - alpha is supported to scale PE before adding to input, default=1.0
        """
        assert (
            isinstance(X, Tensor)
            and X.shape[-2] <= self.max_seq_len
            and X.shape[-1] <= self.max_in_dim
        ), f"shape of tensor X({X.shape}) should be (..., <={self.max_seq_len}, <={self.max_in_dim})"
        # [..., seq_len, in_dim]
        X = X + self.get_buffer("pe")[: X.shape[-2], : X.shape[-1]] * alpha
        return X


class Attention(nn.Module):
    def __init__(
        self,
        in_dim_q: int,
        hid_dim: int,
        out_dim_v: int,
        in_dim_k: Optional[int] = None,
        n_heads: int = 1,
        out_dim_o: Optional[int] = None,
    ):
        """
        Scale-Dot Attention
        - scale factor is 1/sqrt(hid_dim)
        - both self-attention and cross-attention are supported
        - multi-head attention is supported
        ---
        Args:
        - if in_dim_k is None, default = in_dim_q, as self-attention
        - if in_dim_k is not None, as cross-attention
        - if n_heads > 1, a projection layer is added, out_dim = out_dim_o
        - if out_dim_o is None, default = out_dim_v
        """
        super(Attention, self).__init__()
        if in_dim_k is None:
            self.is_cross = False
            self.in_dim_q = self.in_dim_k = in_dim_q
        else:
            self.is_cross = True
            self.in_dim_q, self.in_dim_k = in_dim_q, in_dim_k
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        if n_heads == 1 or out_dim_o is None:
            self.out_dim_v = self.out_dim_o = out_dim_v
        else:
            self.out_dim_v, self.out_dim_o = out_dim_v, out_dim_o
        self.Wq = nn.Linear(self.in_dim_q, self.n_heads * self.hid_dim, bias=False)
        self.Wk = nn.Linear(self.in_dim_k, self.n_heads * self.hid_dim, bias=False)
        self.Wv = nn.Linear(self.in_dim_k, self.n_heads * self.out_dim_v, bias=False)
        self.scale_factor: float = self.hid_dim**-0.5
        if self.n_heads > 1:
            self.Wo = nn.Linear(
                self.n_heads * self.out_dim_v, self.out_dim_o, bias=False
            )

    def forward(
        self,
        Xq: Tensor,
        Xk: Optional[Tensor] = None,
        mask_bef: Optional[Tensor] = None,
        mask_fill_bef: float = -torch.inf,
        mask_aft: Optional[Tensor] = None,
        mask_fill_aft: float = 0,
        need_attention: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Args:
        - Xq: input tensor of query in shape[..., seq_len_q, in_dim_q]
        - Xk: only needed for cross-attention in shape[..., seq_len_k, in_dim_k]
        - mask_x: mask in shape[(batch_size, ..., n_heads,) seq_len_q, seq_len_k] for attention weights
            - dtype:
                - bool: True for masked positions to be filled with mask_fill
                - float: scale factor to be multiplied
            - x: bef for before softmax, aft for after softmax
        - mask_fill_x: value to fill in masked positions where mask=True
        - need_attention (bool): whether need attention
        ---
        Returns:
        - output: [..., seq_len_q, out_dim_o]
        - attention: [..., n_heads, seq_len_q, seq_len_k] if need_attention=True else None
        """
        if self.is_cross:
            assert Xk is not None, f"Xk is required for cross-attention"
        else:
            assert Xk is None, f"Xk is forbidden for self-attention"
            Xk = Xq
        # Xq: [..., seq_len_q, in_dim_q]; Xk/Xv: [..., seq_len_k, in_dim_k]
        Xv = Xk
        Q: Tensor = self.Wq(Xq)  # [..., seq_len_q, n_heads*hid_dim]
        K: Tensor = self.Wk(Xk)  # [..., seq_len_k, n_heads*hid_dim]
        V: Tensor = self.Wv(Xv)  # [..., seq_len_k, n_heads*out_dim_v]
        # [..., n_heads, seq_len_q, hid_dim]
        Q = Q.view(*Q.shape[:-1], self.n_heads, self.hid_dim).transpose(-3, -2)
        # [..., n_heads, seq_len_k, hid_dim]
        K = K.view(*K.shape[:-1], self.n_heads, self.hid_dim).transpose(-3, -2)
        # [..., n_heads, seq_len_k, out_dim_v]
        V = V.view(*V.shape[:-1], self.n_heads, self.out_dim_v).transpose(-3, -2)
        # [..., n_heads, seq_len_q, seq_len_k]
        attention: Tensor = torch.matmul(Q, K.transpose(-2, -1)) * self.scale_factor
        if mask_bef is not None:
            if mask_bef.dtype == torch.bool:
                attention = attention.masked_fill(mask_bef, mask_fill_bef)
            else:
                attention = attention * mask_bef
        attention = torch.softmax(attention, dim=-1)
        if mask_aft is not None:
            if mask_aft.dtype == torch.bool:
                attention = attention.masked_fill(mask_aft, mask_fill_aft)
            else:
                attention = attention * mask_aft
        # [..., seq_len_q, n_heads, out_dim_v]
        output: Tensor = torch.matmul(attention, V).transpose(-3, -2)
        # [..., seq_len_q, n_heads*out_dim_v]
        output = output.reshape(*output.shape[:-2], self.n_heads * self.out_dim_v)
        # [..., seq_len_q, out_dim_o]
        if self.n_heads > 1:
            output = self.Wo(output)
        if not need_attention:
            attention = None
        return output, attention
