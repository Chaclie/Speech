from typing import Any, Optional, Union
import torch
from torch import nn, Tensor

from .layers import PositionalEncoding
from .modules import Coder, TokenPrenet, LinearNet, MelPostnet

from ..util.utils import get_mask_of_lengths, get_mask_of_future


class TransformerTTS(nn.Module):
    def __init__(self, m_cfg: dict[str, dict[str, Any]]):
        super(TransformerTTS, self).__init__()
        set_res, set_hint = self.set_config(m_cfg)
        assert set_res, set_hint
        self.tok_prenet = TokenPrenet(
            n_toks=self.config["tok_prenet"]["n_toks"],
            conv_n_layers=self.config["tok_prenet"]["conv_n_layers"],
            conv_k_size=self.config["tok_prenet"]["conv_k_size"],
            out_dim=self.config["encoder"]["att_in_dim_q"],
            embed_dim=self.config["tok_prenet"]["embed_dim"],
            pad_idx=self.config["tok_prenet"]["pad_idx"],
            conv_hid_dim=self.config["tok_prenet"]["conv_hid_dim"],
        )
        self.mel_prenet = LinearNet(
            n_layers=self.config["mel_prenet"]["n_layers"],
            in_dim=self.config["mel_prenet"]["n_mels"],
            hid_dim=self.config["mel_prenet"]["hid_dim"],
            out_dim=self.config["decoder"]["att_in_dim_q"],
        )
        self.pos_encoding = PositionalEncoding(
            max(
                self.config["tok_prenet"]["max_tok_len"],
                self.config["mel_prenet"]["max_mel_len"],
            ),
            max(
                self.config["encoder"]["att_in_dim_q"],
                self.config["decoder"]["att_in_dim_q"],
            ),
        )
        self.register_parameter("tok_pos_alpha", nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("mel_pos_alpha", nn.Parameter(torch.tensor(1.0)))
        self.encoder = Coder(
            n_blocks=self.config["encoder"]["n_blocks"],
            in_dim=self.config["encoder"]["att_in_dim_q"],
            att_hid_dim=self.config["encoder"]["att_hid_dim"],
            ffn_hid_dim=self.config["encoder"]["ffn_hid_dim"],
            att_n_heads=self.config["encoder"]["att_n_heads"],
            att_in_dim_k=None,
            att_out_dim_v=self.config["encoder"]["att_out_dim_v"],
            norm_first=self.config["encoder"]["norm_first"],
        )
        if self.config["encoder"]["norm_first"]:
            self.enc_norm = nn.LayerNorm(self.config["encoder"]["att_in_dim_q"])
        self.decoder = Coder(
            n_blocks=self.config["decoder"]["n_blocks"],
            in_dim=self.config["decoder"]["att_in_dim_q"],
            att_hid_dim=self.config["decoder"]["att_hid_dim"],
            ffn_hid_dim=self.config["decoder"]["ffn_hid_dim"],
            att_n_heads=self.config["decoder"]["att_n_heads"],
            att_in_dim_k=self.config["encoder"]["att_in_dim_q"],
            att_out_dim_v=self.config["decoder"]["att_out_dim_v"],
            norm_first=self.config["decoder"]["norm_first"],
        )
        if self.config["decoder"]["norm_first"]:
            self.dec_norm = nn.LayerNorm(self.config["decoder"]["att_in_dim_q"])
        self.mel_proj = nn.Linear(
            self.config["decoder"]["att_in_dim_q"], self.config["mel_prenet"]["n_mels"]
        )
        self.mel_postnet = MelPostnet(
            in_dim=self.config["mel_prenet"]["n_mels"],
            conv_n_layers=self.config["mel_postnet"]["conv_n_layers"],
            conv_k_size=self.config["mel_postnet"]["conv_k_size"],
            conv_hid_dim=self.config["mel_postnet"]["conv_hid_dim"],
        )
        self.stp_proj = nn.Linear(self.config["decoder"]["att_in_dim_q"], 1)

    def set_config(self, m_cfg: dict[str, dict[str, Any]]) -> tuple[bool, str]:
        self.config: dict[str, dict[str, Any]] = {}
        if not isinstance(m_cfg, dict):
            return False, "require config as dict"
        attrs: dict[str, dict[str, tuple[type, function, str]]] = {
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
                "conv_k_size": (int, lambda x: x > 0, "int(>0)"),
                "conv_hid_dim": (int, lambda x: x > 0, "int(>0)"),
            },
            "mel_prenet": {
                "max_mel_len": (int, lambda x: x > 0, "int(>0)"),
                "n_mels": (int, lambda x: x > 0, "int(>0)"),
                "n_layers": (int, lambda x: x > 0, "int(>0)"),
                "hid_dim": (int, lambda x: x > 0, "int(>0)"),
            },
            "encoder": {
                "n_blocks": (int, lambda x: x > 0, "int(>0)"),
                "att_n_heads": (int, lambda x: x > 0, "int(>0)"),
                "att_in_dim_q": (int, lambda x: x > 0, "int(>0)"),
                "att_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "att_out_dim_v": (int, lambda x: x > 0, "int(>0)"),
                "ffn_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "norm_first": (bool, lambda x: isinstance(x, bool), "bool"),
            },
            "decoder": {
                "n_blocks": (int, lambda x: x > 0, "int(>0)"),
                "att_n_heads": (int, lambda x: x > 0, "int(>0)"),
                "att_in_dim_q": (int, lambda x: x > 0, "int(>0)"),
                "att_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "att_out_dim_v": (int, lambda x: x > 0, "int(>0)"),
                "ffn_hid_dim": (int, lambda x: x > 0, "int(>0)"),
                "norm_first": (bool, lambda x: isinstance(x, bool), "bool"),
            },
            "mel_postnet": {
                "conv_n_layers": (int, lambda x: x > 0, "int(>0)"),
                "conv_k_size": (int, lambda x: x > 0, "int(>0)"),
                "conv_hid_dim": (int, lambda x: x > 0, "int(>0)"),
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
        tok_lens: Tensor,
        X_tok: Tensor,
        mel_lens: Tensor,
        X_mel: Tensor,
        dropouts: dict[str, float] = {},
        need_attns: dict[str, list[int]] = {},
        need_block_outs: dict[str, list[int]] = {},
        mask_output: bool = True,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        dict[str, dict[int, Tensor]],
        dict[str, dict[int, Tensor]],
    ]:
        """
        Args:
        - tok_lens: [batch_size]
        - X_tok: [batch_size, tok_len]
        - mel_lens: [batch_size]
        - X_mel: [batch_size, mel_len, mel_dim]
        - dropouts: dropout for each module: tok_prenet, mel_prenet, encoder, decoder, mel_postnet
        - need_attns: need_attn for enc_self, dec_self, dec_cros
        - need_block_outs: need_block_out for encoder, decoder
        - mask_output: mask the output according to mel_lens
        ---
        Returns:
        - mel_out/mel_out_post: [batch_size, mel_len, mel_dim]
        - stp_out: [batch_size, mel_len]
        - attns:
            - enc_self: dict of [batch_size, n_heads, tok_len, tok_len]
            - dec_self: dict of [batch_size, n_heads, mel_len, mel_len]
            - dec_cros: dict of [batch_size, n_heads, mel_len, tok_len]
        - block_outs:
            - encoder: dict of [batch_size, tok_len, enc_dim_in_q]
            - decoder: dict of [batch_size, mel_len, dec_dim_in_q]
        """
        tok_len_lim = self.config["tok_prenet"]["max_tok_len"]
        if X_tok.shape[-1] > tok_len_lim:
            X_tok = X_tok[:, :tok_len_lim]  # [batch_size, tok_len]
            tok_lens = torch.where(tok_lens > tok_len_lim, tok_len_lim, tok_lens)
        mel_len_lim = self.config["mel_prenet"]["max_mel_len"]
        if X_mel.shape[-2] > mel_len_lim:
            X_mel = X_mel[:, :mel_len_lim]  # [batch_size, mel_len, mel_dim]
            mel_lens = torch.where(mel_lens > mel_len_lim, mel_len_lim, mel_lens)
        batch_size = X_tok.shape[0]
        max_tok_len, max_mel_len = X_tok.shape[1], X_mel.shape[1]
        # masks
        tok_len_mask = get_mask_of_lengths(tok_lens)  # [batch_size, max_tok_len]
        mel_len_mask = get_mask_of_lengths(mel_lens)  # [batch_size, max_mel_len]
        mel_fut_mask = get_mask_of_future(
            max_mel_len, X_mel.device
        )  # [max_mel_len, max_mel_len]
        # [batch_size, tok_len, enc_dim_in_q]
        X_tok = self.tok_prenet(X_tok, dropouts.get("tok_prenet", 0.0))
        X_tok = self.pos_encoding(X_tok, self.get_parameter("tok_pos_alpha"))
        X_tok, enc_self_attns, _, enc_block_outs = self.encoder(
            X_self=X_tok,
            mask_self_bef=tok_len_mask.view(batch_size, 1, 1, max_tok_len),
            mask_fill_self_bef=-torch.inf,
            mask_self_aft=tok_len_mask.view(batch_size, 1, max_tok_len, 1),
            mask_fill_self_aft=0,
            need_self_attn=need_attns.get("enc_self", []),
            need_block_out=need_block_outs.get("encoder", []),
            dropout=dropouts.get("encoder", 0.0),
        )
        if self.config["encoder"]["norm_first"]:
            X_tok = self.enc_norm(X_tok)
        # [batch_size, mel_len, mel_dim]
        X_mel = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    1,
                    self.config["mel_prenet"]["n_mels"],
                    device=X_mel.device,
                ),
                X_mel[:, :-1, :],
            ],
            dim=1,
        )
        # [batch_size, mel_len, dec_dim_in_q]
        X_mel = self.mel_prenet(X_mel, dropouts.get("mel_prenet", 0.0))
        X_mel = self.pos_encoding(X_mel, self.get_parameter("mel_pos_alpha"))
        X_mel, dec_self_attns, dec_cros_attns, dec_block_outs = self.decoder(
            X_self=X_mel,
            mask_self_bef=mel_len_mask.view(batch_size, 1, 1, max_mel_len)
            | mel_fut_mask.view(1, 1, max_mel_len, max_mel_len),
            mask_fill_self_bef=-torch.inf,
            mask_self_aft=mel_len_mask.view(batch_size, 1, max_mel_len, 1),
            mask_fill_self_aft=0,
            need_self_attn=need_attns.get("dec_self", []),
            X_cros=X_tok,
            mask_cros_bef=tok_len_mask.view(batch_size, 1, 1, max_tok_len),
            mask_fill_cros_bef=-torch.inf,
            mask_cros_aft=mel_len_mask.view(batch_size, 1, max_mel_len, 1),
            mask_fill_cros_aft=0,
            need_cros_attn=need_attns.get("dec_cros", []),
            need_block_out=need_block_outs.get("decoder", []),
            dropout=dropouts.get("decoder", 0.0),
        )
        if self.config["decoder"]["norm_first"]:
            X_mel = self.dec_norm(X_mel)
        # [batch_size, mel_len, mel_dim]
        mel_out: Tensor = self.mel_proj(X_mel)
        mel_out_post: Tensor = mel_out + self.mel_postnet(
            X=mel_out,
            mask=mel_len_mask.view(batch_size, 1, max_mel_len),
            mask_fill=0,
            dropout=dropouts.get("mel_postnet", 0.0),
        )
        # [batch_size, mel_len, 1]
        stp_out: Tensor = self.stp_proj(X_mel)
        stp_out = stp_out.squeeze(-1)  # [batch_size, mel_len]
        attns: dict[str, dict[int, Tensor]] = {
            "enc_self": enc_self_attns,
            "dec_self": dec_self_attns,
            "dec_cros": dec_cros_attns,
        }
        block_outs: dict[str, dict[int, Tensor]] = {
            "encoder": enc_block_outs,
            "decoder": dec_block_outs,
        }
        if mask_output:
            mask = get_mask_of_lengths(mel_lens).unsqueeze(-1)
            mel_out = mel_out.masked_fill(mask, 0.0)
            mel_out_post = mel_out_post.masked_fill(mask, 0.0)
            stp_out = stp_out.masked_fill(mask[:, :, 0], 1e3)
        return mel_out, mel_out_post, stp_out, attns, block_outs

    def synthesize(
        self,
        X_tok: Tensor,
        max_mel_len: int = -1,
        need_attns: dict[str, list[int]] = {},
        need_block_outs: dict[str, list[int]] = {},
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        dict[str, dict[int, Tensor]],
        dict[str, dict[int, Tensor]],
    ]:
        """
        Args:
        - X_tok: [tok_len]
        - max_mel_len:
            max length of synthesized melspectrogram,
            <0 means keeping generating until reaching the limit len of model
        - need_attns: need_attn for enc_self, dec_self, dec_cros
        - need_block_outs: need_block_out for encoder, decoder
        ---
        Returns:
        - mel_out/mel_out_post: [mel_len, mel_dim]
        - stp_out: [mel_len]
        - attns:
            - enc_self: dict of [n_heads, tok_len, tok_len]
            - dec_self: dict of [n_heads, mel_len, mel_len]
            - dec_cros: dict of [n_heads, mel_len, tok_len]
        - block_outs:
            - encoder: dict of [tok_len, enc_dim_in_q]
            - decoder: dict of [mel_len, dec_dim_in_q]
        """
        max_tok_len: int = X_tok.shape[-1]
        tok_len_lim = self.config["tok_prenet"]["max_tok_len"]
        if max_tok_len > tok_len_lim:
            raise ValueError(
                f"length of X_tok({max_tok_len}) out of valid range({tok_len_lim})"
            )
        mel_len_lim = self.config["mel_prenet"]["max_mel_len"]
        if max_mel_len < 0:
            max_mel_len = mel_len_lim
        elif max_mel_len > mel_len_lim:
            raise ValueError(
                f"max_mel_len({max_mel_len}) out of valid range({mel_len_lim})"
            )
        mel_dim = self.config["mel_prenet"]["n_mels"]
        decoder_n_blocks = self.config["decoder"]["n_blocks"]
        decoder_att_n_heads = self.config["decoder"]["att_n_heads"]
        need_attns_dec_self: list[int] = need_attns.get("dec_self", [])
        need_attns_dec_cros: list[int] = need_attns.get("dec_cros", [])
        need_block_outs_dec: list[int] = need_block_outs.get("decoder", [])
        mel_len: int = 0
        mel_out: Tensor = torch.zeros(max_mel_len, mel_dim, device=X_tok.device)
        mel_out_post: Tensor = torch.zeros(
            1, max_mel_len + 1, mel_dim, device=X_tok.device
        )
        stp_out: Tensor = torch.zeros(max_mel_len, device=X_tok.device)
        mel_fut_mask = get_mask_of_future(
            max_mel_len, X_tok.device
        )  # [max_mel_len, max_mel_len]
        dec_self_attns: dict[int, Tensor] = {
            i: torch.zeros(
                decoder_att_n_heads, max_mel_len, max_mel_len, device=X_tok.device
            )
            for i in range(decoder_n_blocks)
            if i in need_attns_dec_self
        }
        dec_cros_attns: dict[int, Tensor] = {
            i: torch.zeros(
                decoder_att_n_heads, max_mel_len, max_tok_len, device=X_tok.device
            )
            for i in range(decoder_n_blocks)
            if i in need_attns_dec_cros
        }
        dec_block_outs: dict[int, Tensor] = {
            i: torch.zeros(
                max_mel_len, self.config["decoder"]["att_in_dim_q"], device=X_tok.device
            )
            for i in range(decoder_n_blocks)
            if i in need_block_outs_dec
        }
        # [1, tok_len, enc_dim_in_q]
        X_tok = self.tok_prenet(X_tok.unsqueeze(0))
        X_tok = self.pos_encoding(X_tok, self.get_parameter("tok_pos_alpha"))
        X_tok, enc_self_attns, _, enc_block_outs = self.encoder(
            X_self=X_tok,
            mask_self_bef=None,
            mask_fill_self_bef=-torch.inf,
            mask_self_aft=None,
            mask_fill_self_aft=0,
            need_self_attn=need_attns.get("enc_self", []),
            need_block_out=need_block_outs.get("encoder", []),
        )
        if self.config["encoder"]["norm_first"]:
            X_tok = self.enc_norm(X_tok)
        enc_self_attns: dict[int, Tensor] = {
            i: attn.squeeze(0) for i, attn in enc_self_attns.items()
        }
        enc_block_outs: dict[int, Tensor] = {
            i: block.squeeze(0) for i, block in enc_block_outs.items()
        }
        for cur_pos in range(0, max_mel_len):
            cur_mel_in: Tensor = mel_out_post[:, : cur_pos + 1]
            # [1, mel_len, dec_dim_in_q]
            cur_mel_in = self.mel_prenet(cur_mel_in)
            cur_mel_in = self.pos_encoding(
                cur_mel_in, self.get_parameter("mel_pos_alpha")
            )
            cur_mel_in, cur_dec_self_attns, cur_dec_cros_attns, cur_dec_block_outs = (
                self.decoder(
                    X_self=cur_mel_in,
                    mask_self_bef=mel_fut_mask[
                        : cur_mel_in.shape[1], : cur_mel_in.shape[1]
                    ],
                    mask_fill_self_bef=-torch.inf,
                    mask_self_aft=None,
                    mask_fill_self_aft=0,
                    need_self_attn=need_attns_dec_self,
                    X_cros=X_tok,
                    mask_cros_bef=None,
                    mask_fill_cros_bef=-torch.inf,
                    mask_cros_aft=None,
                    mask_fill_cros_aft=0,
                    need_cros_attn=need_attns_dec_cros,
                    need_block_out=need_block_outs.get("decoder", []),
                )
            )
            if self.config["decoder"]["norm_first"]:
                cur_mel_in = self.dec_norm(cur_mel_in)
            # [1, mel_len, mel_dim]
            cur_mel_out: Tensor = self.mel_proj(cur_mel_in)
            cur_mel_out_post: Tensor = cur_mel_out + self.mel_postnet(
                X=cur_mel_out, mask=None, mask_fill=0
            )
            # [1, mel_len, 1]
            cur_stp_out: Tensor = self.stp_proj(cur_mel_in)
            for i in range(decoder_n_blocks):
                if i in need_attns_dec_self:
                    dec_self_attns[i][:, cur_pos, : cur_pos + 1] = cur_dec_self_attns[
                        i
                    ][0, :, -1, :]
                if i in need_attns_dec_cros:
                    dec_cros_attns[i][:, cur_pos, :] = cur_dec_cros_attns[i][
                        0, :, -1, :
                    ]
                if i in need_block_outs_dec:
                    dec_block_outs[i][cur_pos, :] = cur_dec_block_outs[i][0, -1, :]
            mel_out[cur_pos, :] = cur_mel_out[0, -1, :]
            mel_out_post[:, cur_pos + 1, :] = cur_mel_out_post[:, -1, :]
            stp_out[cur_pos] = cur_stp_out[0, -1, 0]
            mel_len = cur_pos + 1
            if stp_out[cur_pos] > 0:  # same as torch.sigmoid(stp_out[cur_pos]) > 0.5
                break
        mel_out = mel_out[:mel_len]
        mel_out_post = mel_out_post[0, 1 : mel_len + 1]
        stp_out = stp_out[:mel_len]
        for i in range(decoder_n_blocks):
            if i in need_attns_dec_self:
                dec_self_attns[i] = dec_self_attns[i][:, :mel_len, :mel_len]
            if i in need_attns_dec_cros:
                dec_cros_attns[i] = dec_cros_attns[i][:, :mel_len, :]
            if i in need_block_outs_dec:
                dec_block_outs[i] = dec_block_outs[i][:mel_len]
        attns: dict[str, dict[int, Tensor]] = {
            "enc_self": enc_self_attns,
            "dec_self": dec_self_attns,
            "dec_cros": dec_cros_attns,
        }
        block_outs: dict[str, dict[int, Tensor]] = {
            "encoder": enc_block_outs,
            "decoder": dec_block_outs,
        }
        return mel_out, mel_out_post, stp_out, attns, block_outs


class TransformerTTSLoss(nn.Module):
    def __init__(self, m_cfg: dict[str, dict[str, Any]]):
        super(TransformerTTSLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bcl = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_outputs: tuple[Tensor, Tensor, Tensor],
        grnd_targets: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        - pred_outputs:
            - mel_out/mel_out_post: [batch_size, mel_len, mel_dim]
            - stp_out: [batch_size, mel_len]
        - grnd_targets:
            - mel_lens: [batch_size]
            - mel_grnd: [batch_size, mel_len, mel_dim]
            - stp_grnd: [batch_size, mel_len]
        ---
        Returns:
        - tot_loss: total loss = mel_loss + stp_loss
        - mel_loss: mel loss (mean squared error)
        - stp_loss: stop loss (binary cross entropy with logits)
        """
        mel_out, mel_out_post, stp_out = pred_outputs
        mel_lens, mel_grnd, stp_grnd = grnd_targets
        mel_lens.requires_grad_(False)
        mel_grnd.requires_grad_(False)
        stp_grnd.requires_grad_(False)
        mel_loss = self.mse(mel_out, mel_grnd) + self.mse(mel_out_post, mel_grnd)
        stp_loss = self.bcl(stp_out, stp_grnd)
        tot_loss = mel_loss + stp_loss
        return tot_loss, mel_loss, stp_loss
