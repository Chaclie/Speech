from typing import Union, Optional
import torch
from torch import nn, Tensor

from .layers import PositionalEncoding, Attention
from .blocks import TransBlock, FFTBlock
from ..util.utils import get_mask_of_lengths


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


class Conv1dProj(nn.Module):
    def __init__(
        self,
        in_dim: int,
        conv_n_layers: int,
        conv_k_size: int,
        conv_hid_dim: int,
        out_dim: int,
        proj_first: bool = False,
    ):
        """
        - proj_first:
            - True: in_dim -> out_dim -> [Conv1d -> BatchNorm -> Tanh] * (conv_n_layers-1) -> out_dim
            - False: in_dim -> [Conv1d -> BatchNorm -> Tanh] * (conv_n_layers-1) -> in_dim -> out_dim
        - if in_dim==out_dim, projection is not applied
        - residual connection is applied around conv
        """
        super(Conv1dProj, self).__init__()
        assert conv_k_size % 2 == 1, f"conv_k_size({conv_k_size}) should be odd"
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        (out_dim if proj_first else in_dim) if i == 0 else conv_hid_dim,
                        (
                            (out_dim if proj_first else in_dim)
                            if i == conv_n_layers - 1
                            else conv_hid_dim
                        ),
                        conv_k_size,
                        padding=(conv_k_size - 1) // 2,
                    ),
                    nn.BatchNorm1d(
                        (out_dim if proj_first else in_dim)
                        if i == conv_n_layers - 1
                        else conv_hid_dim
                    ),
                )
                for i in range(conv_n_layers)
            ]
        )
        if in_dim != out_dim:
            self.proj = nn.Linear(in_features=in_dim, out_features=out_dim)
            self.apply_proj = True
        else:
            self.apply_proj = False
        self.proj_first = proj_first

    def forward(
        self,
        X: Tensor,
        mask: Optional[Tensor] = None,
        mask_fill: float = 0.0,
        dropout: float = 0.0,
        need_before_residual: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        [batch_size, seq_len, in_dim]--->[batch_size, seq_len, out_dim]
        - mask: [batch_size, seq_len] as bool
        """
        if mask is not None:
            mask = mask.unsqueeze(-2)
        if self.apply_proj and self.proj_first:
            X = self.proj(X)  # [batch_size, seq_len, out_dim]
        # [batch_size, out_dim if proj_first else in_dim, seq_len]
        X1 = X.transpose(-2, -1)
        for i, conv in enumerate(self.convs):
            # [batch_size, conv_hid_dim, seq_len]
            if mask is not None:
                X1 = X1.masked_fill(mask, mask_fill)
            X1 = conv(X1)
            if i != len(self.convs) - 1:
                X1 = torch.tanh(X1)
            X1 = torch.dropout(X1, dropout, self.training)
        # [batch_size, seq_len, out_dim if proj_first else in_dim]
        X1 = X + X1.transpose(-2, -1)
        if self.apply_proj and not self.proj_first:
            X1 = self.proj(X1)  # [batch_size, seq_len, out_dim]
        if not need_before_residual:
            X = None
        return X1, X


class FFTCoder(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        in_dim: int,
        att_hid_dim: int,
        conv_hid_dim: int,
        conv_k_size: int,
        att_n_heads: int = 1,
        att_in_dim_k: Optional[int] = None,
        att_out_dim_v: Optional[int] = None,
        norm_first: bool = True,
    ):
        """n_blocks FFTBlock"""
        super(FFTCoder, self).__init__()
        self.n_blocks = n_blocks
        self.blocks: list[FFTBlock] = nn.ModuleList(
            [
                FFTBlock(
                    in_dim=in_dim,
                    att_hid_dim=att_hid_dim,
                    conv_hid_dim=conv_hid_dim,
                    conv_k_size=conv_k_size,
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
        Xq: Tensor,
        Xk: Optional[Tensor] = None,
        mask_bef: Optional[Tensor] = None,
        mask_fill_bef: float = -torch.inf,
        mask_aft: Optional[Tensor] = None,
        mask_fill_aft: float = 0,
        need_attn: list[int] = [],
        need_block_out: list[int] = [],
        dropout: float = 0.0,
    ) -> tuple[Tensor, dict[int, Tensor], dict[int, Tensor]]:
        """
        Args similar to FFTBlock:
        - need_attn: indexes(0-based) of blocks of which attention is needed
        - need_block_out: indexes(0-based) of blocks of which output is needed
        ---
        Returns:
        - X: [..., seq_len_q, in_dim_q]
        - dict of attn: [..., n_heads, seq_len_q, seq_len_q]
        - dict of block_out: [..., seq_len_q, in_dim_q]
        """
        attn_dict: dict[int, Tensor] = {}
        block_out_dict: dict[int, Tensor] = {}
        for i, block in enumerate(self.blocks):
            Xq, attn = block(
                Xq=Xq,
                Xk=Xk,
                mask_bef=mask_bef,
                mask_fill_bef=mask_fill_bef,
                mask_aft=mask_aft,
                mask_fill_aft=mask_fill_aft,
                need_attn=i in need_attn,
                dropout=dropout,
            )
            if i in need_attn:
                attn_dict[i] = attn
            if i in need_block_out:
                block_out_dict[i] = Xq
        return Xq, attn_dict, block_out_dict


class Vformer(nn.Module):
    """
    - X_A ---encoder--> X'_A ---len_predictor--> X_len_B
    - X_len_B, X'_A ---seq_translator--> X'_B ---decoder--> X_B
    """

    def __init__(
        self,
        len_limit: tuple[int, int],
        seq_dim: tuple[int, int],
        cd_n_blocks: tuple[int, int],
        cd_att_hid_dim: tuple[int, int],
        cd_conv_hid_dim: tuple[int, int],
        cd_conv_k_size: tuple[int, int],
        lr_att_hid_dim: tuple[int, int],
        lr_conv_hid_dim: tuple[int, int],
        lr_conv_k_size: tuple[int, int],
        cd_att_n_heads: tuple[int, int] = (1, 1),
        lr_att_n_heads: tuple[int, int] = (1, 1),
        cd_att_out_dim_v: tuple[Optional[int], Optional[int]] = (None, None),
        lr_att_out_dim_v: tuple[Optional[int], Optional[int]] = (None, None),
        cd_norm_first: tuple[bool, bool] = (True, True),
        lr_norm_first: tuple[bool, bool] = (True, True),
    ):
        """
        - len_limit: max length limit for sequence A,B
        - seq_dim: feature dim size for sequence A,B
        - cd_xxx: config for encoder and decoder, both of FFTCoder type
        - lr_xxx: config for len_predictor and seq_translator, both of FFTBlock type
        """
        super(Vformer, self).__init__()
        self.len_limit, self.seq_dim = len_limit, seq_dim
        self.pos_encoding = PositionalEncoding(
            max_seq_len=max(len_limit), max_in_dim=max(seq_dim)
        )
        self.encoder, self.decoder = [
            FFTCoder(
                n_blocks=cd_n_blocks[i],
                in_dim=seq_dim[i],
                att_hid_dim=cd_att_hid_dim[i],
                conv_hid_dim=cd_conv_hid_dim[i],
                conv_k_size=cd_conv_k_size[i],
                att_n_heads=cd_att_n_heads[i],
                att_in_dim_k=None,  # seq_dim[1 - i] if i else None,
                att_out_dim_v=cd_att_out_dim_v[i],
                norm_first=cd_norm_first[i],
            )
            for i in range(2)
        ]
        self.len_predictor, self.seq_translator = [
            FFTBlock(
                in_dim=seq_dim[1] if i else 1,
                att_hid_dim=lr_att_hid_dim[i],
                conv_hid_dim=lr_conv_hid_dim[i],
                conv_k_size=lr_conv_k_size[i],
                att_n_heads=lr_att_n_heads[i],
                att_in_dim_k=seq_dim[0],
                att_out_dim_v=lr_att_out_dim_v[i],
                norm_first=lr_norm_first[i],
            )
            for i in range(2)
        ]

    def forward(
        self,
        X_A: Tensor,
        A_lens: Tensor,
        B_lens: Optional[Tensor] = None,
        cd_need_attns: tuple[list[int], list[int]] = ([], []),
        cd_need_block_outs: tuple[list[int], list[int]] = ([], []),
        cd_dropouts: tuple[float, float] = (0.0, 0.0),
        lr_need_attns: tuple[bool, bool] = (False, False),
        lr_dropouts: tuple[float, float] = (0.0, 0.0),
        pos_enc_input: bool = True,
    ) -> tuple[
        Tensor,
        Tensor,
        dict[str, dict[int, Tensor]],
        dict[str, dict[int, Tensor]],
        dict[str, Tensor],
    ]:
        """
        Args:
        - X_A: [batch_size, seq_len_A, seq_dim_A], seq_len_A>len_limit_A will result in truncation
        - A_lens: [batch_size]
        - B_lens: [batch_size], X_B will be predicted according to B_lens when it is not None
        - cd_xxx: config for encoder and decoder
            - ~_need_attns: indexes(0-based) of blocks of which attention is needed
            - ~_need_block_outs: indexes(0-based) of blocks of which output is needed, -1 for requiring the input of the first block
            - ~_dropouts: dropout probability
        - lr_xxx: config for len_predictor and seq_translator
            - ~_need_attns: whether attention is needed
            - ~_dropouts: dropout probability
        - pos_enc_input: whether to add positional encoding to X_A before encoder
        ---
        Returns:
        - X_B: [batch_size, seq_len_B, seq_dim_B]
        - B_lens: [batch_size] used for mask
        - B_lens_pred: [batch_size]
        - cd_attns: attns from encoder and decoder
        - cd_block_outs: block outputs from encoder and decoder
        - lr_attns: attns from len_predictor and seq_translator
        """
        batch_size: int = X_A.shape[0]
        if X_A.shape[1] > self.len_limit[0]:
            X_A = X_A[:, self.len_limit[0]]
            A_lens = torch.clip(A_lens, max=self.len_limit[0])
        A_max_len: int = X_A.shape[1]
        A_len_mask: Tensor = get_mask_of_lengths(A_lens)  # [batch_size, seq_len_A]
        # =====Encode=====
        if pos_enc_input:
            X_A = self.pos_encoding(X_A)
        # X_out/enc_block_out: [batch_size, seq_len_A, seq_dim_A]
        # enc_attn: [batch_size, n_heads, seq_len_A, seq_len_A]
        X_out, enc_attns, enc_block_outs = self.encoder(
            Xq=X_A,
            Xk=None,
            mask_bef=A_len_mask.view(batch_size, 1, 1, A_max_len),
            mask_fill_bef=-torch.inf,
            mask_aft=A_len_mask.view(batch_size, 1, A_max_len, 1),
            mask_fill_aft=0,
            need_attn=cd_need_attns[0],
            need_block_out=cd_need_block_outs[0],
            dropout=cd_dropouts[0],
        )
        if -1 in cd_need_block_outs[0]:
            enc_block_outs[-1] = X_A
        X_A = X_out
        # =====Translate=====
        # B_lens_pred: [batch_size, 1, 1]
        # len_attn: [batch_size, n_heads, 1, seq_len_A]
        B_lens_pred, len_attn = self.len_predictor(
            Xq=torch.ones(batch_size, 1, 1, device=X_A.device),
            Xk=X_A,
            mask_bef=A_len_mask.view(batch_size, 1, 1, A_max_len),
            mask_fill_bef=-torch.inf,
            mask_aft=None,
            mask_fill_aft=0,
            need_attn=lr_need_attns[0],
            dropout=lr_dropouts[0],
        )
        B_lens_pred: Tensor = B_lens_pred.squeeze(-2).squeeze(-1)  # [batch_size]
        if B_lens is None:
            B_lens = torch.clip(
                torch.exp(B_lens_pred), min=1, max=self.len_limit[1]
            ).to(dtype=torch.int)
        else:
            B_lens = torch.clip(B_lens, max=self.len_limit[1])
        B_max_len: int = torch.max(B_lens).item()
        B_len_mask: Tensor = get_mask_of_lengths(B_lens)  # [batch_size, seq_len_B]
        X_B: Tensor = self.pos_encoding(
            torch.zeros(batch_size, B_max_len, self.seq_dim[1], device=X_A.device)
        )  # [batch_size, seq_len_B, seq_dim_B]
        # seq_attn: [batch_size, n_heads, seq_len_B, seq_len_A]
        X_B, seq_attn = self.seq_translator(
            Xq=X_B,
            Xk=X_A,
            mask_bef=A_len_mask.view(batch_size, 1, 1, A_max_len),
            mask_fill_bef=-torch.inf,
            mask_aft=B_len_mask.view(batch_size, 1, B_max_len, 1),
            mask_fill_aft=0,
            need_attn=lr_need_attns[1],
            dropout=lr_dropouts[1],
        )
        # X_B = self.pos_encoding(X_B)
        # =====Decode=====
        # X_B/dec_block_out: [batch_size, seq_len_B, seq_dim_B]
        # dec_attn: [batch_size, n_heads, seq_len_B, seq_len_B]
        X_out, dec_attns, dec_block_outs = self.decoder(
            Xq=X_B,
            # Xk=X_A,
            # mask_bef=A_len_mask.view(batch_size, 1, 1, A_max_len),
            Xk=None,
            mask_bef=B_len_mask.view(batch_size, 1, 1, B_max_len),
            mask_fill_bef=-torch.inf,
            mask_aft=B_len_mask.view(batch_size, 1, B_max_len, 1),
            mask_fill_aft=0,
            need_attn=cd_need_attns[1],
            need_block_out=cd_need_block_outs[1],
            dropout=cd_dropouts[1],
        )
        if -1 in cd_need_block_outs[1]:
            dec_block_outs[-1] = X_B
        X_B = X_out
        # =====Return=====
        cd_attns: dict[str, dict[int, Tensor]] = {
            "encoder": enc_attns,
            "decoder": dec_attns,
        }
        cd_block_outs: dict[str, dict[int, Tensor]] = {
            "encoder": enc_block_outs,
            "decoder": dec_block_outs,
        }
        lr_attns: dict[str, Tensor] = {
            "len_predictor": len_attn,
            "seq_translator": seq_attn,
        }
        return X_B, B_lens, B_lens_pred, cd_attns, cd_block_outs, lr_attns
