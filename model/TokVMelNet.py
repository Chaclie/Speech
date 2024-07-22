from typing import Any, Optional, Union, Callable
import torch
from torch import nn, Tensor

from .modules import Vformer, Conv1dProj

from ..util.utils import get_mask_of_lengths


class TokVMelNet(nn.Module):
    def __init__(self, m_cfg: dict[str, dict[str, Any]]):
        super(TokVMelNet, self).__init__()
        # =====Config=====
        set_res, set_hint = self.set_config(m_cfg)
        assert set_res, set_hint
        tok_cod_cfg = self.config["tok_coder"]
        mel_cod_cfg = self.config["mel_coder"]
        t2m_lpd_cfg = self.config["tok2mel_lpd"]
        t2m_stl_cfg = self.config["tok2mel_stl"]
        tok_pre_cfg = self.config["tok_prenet"]
        mel_pst_cfg = self.config["mel_postnet"]
        # =====Model=====
        self.embedding = nn.Embedding(
            num_embeddings=tok_pre_cfg["n_toks"],
            embedding_dim=tok_pre_cfg["embed_dim"],
            padding_idx=tok_pre_cfg["pad_idx"],
        )
        self.vformer = Vformer(
            len_limit=(tok_pre_cfg["max_tok_len"], mel_pst_cfg["max_mel_len"]),
            seq_dim=(tok_cod_cfg["att_in_dim_q"], mel_cod_cfg["att_in_dim_q"]),
            cd_n_blocks=(tok_cod_cfg["n_blocks"], mel_cod_cfg["n_blocks"]),
            cd_att_hid_dim=(tok_cod_cfg["att_hid_dim"], mel_cod_cfg["att_hid_dim"]),
            cd_conv_hid_dim=(tok_cod_cfg["conv_hid_dim"], mel_cod_cfg["conv_hid_dim"]),
            cd_conv_k_size=(tok_cod_cfg["conv_k_size"], mel_cod_cfg["conv_k_size"]),
            lr_att_hid_dim=(t2m_lpd_cfg["att_hid_dim"], t2m_stl_cfg["att_hid_dim"]),
            lr_conv_hid_dim=(t2m_lpd_cfg["conv_hid_dim"], t2m_stl_cfg["conv_hid_dim"]),
            lr_conv_k_size=(t2m_lpd_cfg["conv_k_size"], t2m_stl_cfg["conv_k_size"]),
            cd_att_n_heads=(tok_cod_cfg["att_n_heads"], mel_cod_cfg["att_n_heads"]),
            lr_att_n_heads=(t2m_lpd_cfg["att_n_heads"], t2m_stl_cfg["att_n_heads"]),
            cd_att_out_dim_v=(
                tok_cod_cfg["att_out_dim_v"],
                mel_cod_cfg["att_out_dim_v"],
            ),
            lr_att_out_dim_v=(
                t2m_lpd_cfg["att_out_dim_v"],
                t2m_stl_cfg["att_out_dim_v"],
            ),
            cd_norm_first=(tok_cod_cfg["norm_first"], mel_cod_cfg["norm_first"]),
            lr_norm_first=(t2m_lpd_cfg["norm_first"], t2m_stl_cfg["norm_first"]),
        )
        self.tok_prenet = Conv1dProj(
            in_dim=tok_pre_cfg["embed_dim"],
            conv_n_layers=tok_pre_cfg["conv_n_layers"],
            conv_k_size=tok_pre_cfg["conv_k_size"],
            conv_hid_dim=tok_pre_cfg["conv_hid_dim"],
            out_dim=tok_cod_cfg["att_in_dim_q"],
            proj_first=tok_pre_cfg["proj_first"],
        )
        self.mel_postnet = Conv1dProj(
            in_dim=mel_cod_cfg["att_in_dim_q"],
            conv_n_layers=mel_pst_cfg["conv_n_layers"],
            conv_k_size=mel_pst_cfg["conv_k_size"],
            conv_hid_dim=mel_pst_cfg["conv_hid_dim"],
            out_dim=mel_pst_cfg["n_mels"],
            proj_first=mel_pst_cfg["proj_first"],
        )

    def set_config(self, m_cfg: dict[str, dict[str, Any]]) -> tuple[bool, str]:
        self.config: dict[str, dict[str, Any]] = {}
        if not isinstance(m_cfg, dict):
            return False, "require config as dict"
        attrs: dict[str, dict[str, tuple[type, function, str]]] = {
            "tok_coder": {
                "n_blocks": (int, lambda x: x > 0, "int(>0)"),
                "att_n_heads": (int, lambda x: x > 0, "int(>0)"),
                "att_in_dim_q": (int, lambda x: x > 0, "int(>0)"),
                "att_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "att_out_dim_v": (int, lambda x: x > 0, "int(>0)"),
                "conv_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "conv_k_size": (int, lambda x: x > 0 and x % 2 == 1, "odd int(>0)"),
                "norm_first": (bool, lambda x: isinstance(x, bool), "bool"),
            },
            "mel_coder": {
                "n_blocks": (int, lambda x: x > 0, "int(>0)"),
                "att_n_heads": (int, lambda x: x > 0, "int(>0)"),
                "att_in_dim_q": (int, lambda x: x > 0, "int(>0)"),
                "att_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "att_out_dim_v": (int, lambda x: x > 0, "int(>0)"),
                "conv_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "conv_k_size": (int, lambda x: x > 0 and x % 2 == 1, "odd int(>0)"),
                "norm_first": (bool, lambda x: isinstance(x, bool), "bool"),
            },
            "tok2mel_lpd": {
                "att_n_heads": (int, lambda x: x > 0, "int(>0)"),
                "att_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "att_out_dim_v": (int, lambda x: x > 0, "int(>0)"),
                "conv_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "conv_k_size": (int, lambda x: x > 0 and x % 2 == 1, "odd int(>0)"),
                "norm_first": (bool, lambda x: isinstance(x, bool), "bool"),
            },
            "tok2mel_stl": {
                "att_n_heads": (int, lambda x: x > 0, "int(>0)"),
                "att_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "att_out_dim_v": (int, lambda x: x > 0, "int(>0)"),
                "conv_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "conv_k_size": (int, lambda x: x > 0 and x % 2 == 1, "odd int(>0)"),
                "norm_first": (bool, lambda x: isinstance(x, bool), "bool"),
            },
            "tok_prenet": {
                "max_tok_len": (int, lambda x: x > 0, "int(>0)"),
                "n_toks": (int, lambda x: x > 0, "int(>0)"),
                "embed_dim": (int, lambda x: x > 0, "int(>0)"),
                "pad_idx": (
                    Optional[int],
                    lambda x: x is None or x >= 0,
                    "Optional[int(>=0)]",
                ),
                "conv_n_layers": (int, lambda x: x > 0, "int(>0)"),
                "conv_k_size": (int, lambda x: x > 0 and x % 2 == 1, "odd int(>0)"),
                "conv_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "proj_first": (bool, lambda x: isinstance(x, bool), "bool"),
            },
            "mel_postnet": {
                "max_mel_len": (int, lambda x: x > 0, "int(>0)"),
                "n_mels": (int, lambda x: x > 0, "int(>0)"),
                "conv_n_layers": (int, lambda x: x > 0, "int(>0)"),
                "conv_k_size": (int, lambda x: x > 0 and x % 2 == 1, "odd int(>0)"),
                "conv_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "proj_first": (bool, lambda x: isinstance(x, bool), "bool"),
            },
        }
        for module_name, require_cfg in attrs.items():
            module_cfg = m_cfg.get(module_name, None)
            if isinstance(module_cfg, dict):
                for attr_name, (
                    attr_type,
                    attr_check,
                    attr_hint,
                ) in require_cfg.items():
                    attr_val = module_cfg.get(attr_name, None)
                    if not isinstance(attr_val, attr_type) or not attr_check(attr_val):
                        return (
                            False,
                            f"require {module_name}.{attr_name} as {attr_hint}",
                        )
                self.config[module_name] = {
                    attr_name: module_cfg[attr_name] for attr_name in require_cfg.keys()
                }
            else:
                return False, f"require {module_name} config as dict"
        return True, "valid config"

    def forward(
        self,
        X_A: Tensor,
        A_lens: Tensor,
        B_lens: Optional[Tensor] = None,
        A_pre_dropout: float = 0.0,
        B_pst_dropout: float = 0.0,
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
        Tensor,
    ]:
        """
        Args:
        - similar to Vformer.forward, A for tok, B for mel
        - X_A:
            - [batch_size, tok_len] will be embedded first
            - xx_len>len_limit will result in truncation
        - A_lens: [batch_size]
        - B_lens: [batch_size], X_B will be predicted according to B_lens when it is not None
        - A_pre_dropout: dropout probability for prenet of X_A
        - B_pst_dropout: dropout probability for postnet of X_B
        - cd_xxx: config for encoder or decoder
            - ~_need_attns: indexes(0-based) of blocks of which attention is needed
            - ~_need_block_outs: indexes(0-based) of blocks of which output is needed, -1 for requiring the input of the first block
            - ~_dropouts: dropout probability
        - lr_xxx: config for len_predictor and seq_translator
            - ~_need_attns: whether attention is needed
            - ~_dropouts: dropout probability
        - pos_enc_input: whether to add positional encoding to X_A after prenet and before encoder
        ---
        Returns:
        - X_B: [batch_size, seq_len_B, seq_dim_B]
        - B_lens_pred: [batch_size]
        - cd_attns: attns from encoder and decoder
        - cd_block_outs: block outputs from encoder and decoder
        - lr_attns: attns from len_predictor and seq_translator
        - X_B_bef_res: input before conv residual connection in Conv1dProj
        """
        A_len_mask: Tensor = get_mask_of_lengths(A_lens)  # [batch_size, seq_len_A]
        X_A = self.embedding(X_A)
        X_A, _ = self.tok_prenet.forward(
            X=X_A,
            mask=A_len_mask,
            mask_fill=0.0,
            dropout=A_pre_dropout,
            need_before_residual=False,
        )
        X_B, B_lens_for_mask, *others = self.vformer(
            X_A=X_A,
            A_lens=A_lens,
            B_lens=B_lens,
            cd_need_attns=cd_need_attns,
            cd_need_block_outs=cd_need_block_outs,
            cd_dropouts=cd_dropouts,
            lr_need_attns=lr_need_attns,
            lr_dropouts=lr_dropouts,
            pos_enc_input=pos_enc_input,
        )
        # [batch_size, seq_len_B]
        B_len_mask: Tensor = get_mask_of_lengths(B_lens_for_mask)
        X_B, X_B_bef_res = self.mel_postnet(
            X=X_B,
            mask=B_len_mask,
            mask_fill=0.0,
            dropout=B_pst_dropout,
            need_before_residual=True,
        )
        X_B = X_B.masked_fill(B_len_mask.unsqueeze(-1), 0.0)
        return X_B, *others, X_B_bef_res


class TokVMelNetLoss(nn.Module):
    def __init__(self, m_cfg: dict[str, dict[str, Any]]):
        super(TokVMelNetLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_outputs: tuple[Tensor, Tensor, Tensor],
        grnd_targets: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        - pred_outputs:
            - mel_lens_pred: [batch_size]
            - mel_pred/mel_bef_res: [batch_size, mel_len, mel_dim]
        - grnd_targets:
            - mel_lens_grnd: [batch_size]
            - mel_grnd: [batch_size, mel_len, mel_dim]
        ---
        Returns:
        - tot_loss: sum of loss below
        - mel_lens_loss: mse for mel lens prediction(in log scale)
        - mel_loss: mse for mel prediction
        - mel_loss_bef_res: mse for mel prediction before residual connection in postnet
        """
        mel_lens_pred, mel_pred, mel_bef_res = pred_outputs
        mel_lens_grnd, mel_grnd = grnd_targets
        mel_grnd.requires_grad_(False)
        mel_lens_grnd.requires_grad_(False)
        mel_lens_loss: Tensor = self.mse(
            mel_lens_pred, torch.log(mel_lens_grnd.float())
        )
        mel_loss: Tensor = self.mse(mel_pred, mel_grnd)
        mel_loss_bef_res: Tensor = self.mse(mel_bef_res, mel_grnd)
        tot_loss: Tensor = mel_lens_loss + mel_loss + mel_loss_bef_res
        return tot_loss, mel_lens_loss, mel_loss, mel_loss_bef_res


def get_model_input(
    batch: tuple[Tensor, Tensor, Tensor, Tensor],
    r_cfg: dict[str, dict[str, Any]],
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    float,
    float,
    tuple[list[int], list[int]],
    tuple[list[int], list[int]],
    tuple[float, float],
    tuple[bool, bool],
    tuple[float, float],
    bool,
]:
    """
    Args:
    - batch:
        - tokids_lens: [batch_size]
        - pad_tokids: [batch_size, tok_len]
        - mel_lens: [batch_size]
        - pad_mel: [batch_size, mel_len, mel_dim]
    - r_cfg:
        - dropout:
            - float: tok_prenet, tok_coder
            - float: mel_coder, mel_postnet
        - need_attns:
            - list[int]: tok_coder, mel_coder
            - bool: tok2mel_len_predictor, tok2mel_seq_translator
        - need_block_outs: list[int]: tok_coder, mel_coder
    ---
    Returns:
    - X_A [batch_size, tok_len]
    - A_lens [batch_size]; B_lens [batch_size]
    - A_pre_dropout, B_pst_dropout,
    - cd_need_attns, cd_need_block_outs both tuple[list[int], list[int]]
    - cd_dropouts tuple[float, float]
    - lr_need_attns tuple[bool, bool]
    - lr_dropouts tuple[float, float]
    - pos_enc_input bool
    """
    tokids_lens, pad_tokids, mel_lens, pad_mel = batch
    X_A: Tensor = pad_tokids
    A_lens: Tensor = tokids_lens
    B_lens: Tensor = mel_lens
    dropouts: dict[str, float] = r_cfg["dropout"]
    need_attns: dict[str, Any] = r_cfg["need_attns"]
    need_block_outs: dict[str, list[int]] = r_cfg["need_block_outs"]
    A_pre_dropout: float = dropouts.get("tok_prenet", 0.0)
    B_pst_dropout: float = dropouts.get("mel_postnet", 0.0)
    cd_need_attns: tuple[list[int], list[int]] = (
        need_attns.get("tok_coder", []),
        need_attns.get("mel_coder", []),
    )
    cd_need_block_outs: tuple[list[int], list[int]] = (
        need_block_outs.get("tok_coder", []),
        need_block_outs.get("mel_coder", []),
    )
    cd_dropouts: tuple[float, float] = (
        dropouts.get("tok_coder", 0.0),
        dropouts.get("mel_coder", 0.0),
    )
    lr_need_attns: tuple[bool, bool] = (
        need_attns.get("tok2mel_len_predictor", False),
        need_attns.get("tok2mel_seq_translator", False),
    )
    lr_dropouts: tuple[float, float] = (
        dropouts.get("tok2mel_len_predictor", 0.0),
        dropouts.get("tok2mel_seq_translator", 0.0),
    )
    pos_enc_input: bool = True
    return (
        X_A,
        A_lens,
        B_lens,
        A_pre_dropout,
        B_pst_dropout,
        cd_need_attns,
        cd_need_block_outs,
        cd_dropouts,
        lr_need_attns,
        lr_dropouts,
        pos_enc_input,
    )


def get_assess_input(
    batch: tuple[Tensor, Tensor, Tensor, Tensor],
    model_output: tuple[
        Tensor,
        Tensor,
        dict[str, dict[int, Tensor]],
        dict[str, dict[int, Tensor]],
        dict[str, Tensor],
        Tensor,
    ],
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor, dict[str, dict[Any, tuple[Tensor, Tensor]]]],
    tuple[Tensor, Tensor, Tensor, Tensor],
]:
    """
    Args:
    - batch:
        - tokids_lens: [batch_size]
        - pad_tokids: [batch_size, tok_len]
        - mel_lens: [batch_size]
        - pad_mel: [batch_size, mel_len, mel_dim]
    - model_output:
        - X_B: [batch_size, mel_len, mel_dim]
        - B_lens_pred: [batch_size]
        - cd_attns: attns from encoder and decoder
        - cd_block_outs: block outputs from encoder and decoder
        - lr_attns: attns from len_predictor and seq_translator
        - X_B_bef_res: input before conv residual connection in Conv1dProj
    ---
    Returns:
    - pred_outputs: mel_lens_pred, mel_pred, mel_bef_res
    - grnd_targets: mel_lens_grnd, mel_grnd
    """
    tokids_lens, pad_tokids, mel_lens, pad_mel = batch
    mel_pred, mel_lens_pred, cd_attns, cd_block_outs, lr_attns, mel_bef_res = (
        model_output
    )
    return tuple([mel_lens_pred, mel_pred, mel_bef_res]), tuple([mel_lens, pad_mel])


def backward(assess_output: tuple[Tensor, Tensor, Tensor, Tensor]):
    tot_loss = assess_output[0]
    tot_loss.backward()


def get_logger_input(
    assess_output: tuple[Tensor, Tensor, Tensor, Tensor]
) -> dict[str, float]:
    """
    Args:
    - assess_output: tot_loss, mel_lens_loss, mel_loss, mel_loss_bef_res
    ---
    Returns:
    - logger_input: tot_loss, mel_loss, mel_lens_loss
    """
    tot_loss, mel_lens_loss, mel_loss, mel_loss_bef_res = assess_output
    logger_input: dict[str, float] = {
        "tot_loss": tot_loss.item(),
        "mel_lens_loss": mel_lens_loss.item(),
        "mel_loss": mel_loss.item(),
        "mel_loss_bef_res": mel_loss_bef_res.item(),
    }
    return logger_input
