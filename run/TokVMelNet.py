import os, json
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Any, Optional
import pypinyin
import numpy as np

from ..util.utils import (
    get_vivid_str,
    tokens_to_indexes,
    indexes_to_tokens,
    to_device,
    mels_to_wave,
    save_wave,
    show_data,
    tensor2array,
)
from ..util.textparser import (
    valid_tokens,
    _unk as unk_token,
    replace_puncts,
    merge_tokens,
    pinyins_to_tokens,
)
from ..util.scheduler import Scheduler
from ..util.runhelper import train_wrapper, contrast_wrapper
from ..model.TokVMelNet import (
    TokVMelNet,
    TokVMelNetLoss,
    get_model_input,
    get_assess_input,
    backward,
    get_logger_input,
)


def train(
    r_cfg: dict[str, dict[str, Any]],
    model: TokVMelNet,
    assess: TokVMelNetLoss,
    optim: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    scheduler: Optional[Scheduler] = None,
    cur_step: int = 1,
    cur_loss: float = float("inf"),
    bes_loss: float = float("inf"),
):
    train_wrapper(
        r_cfg,
        model,
        assess,
        optim,
        train_loader,
        valid_loader,
        get_model_input,
        get_assess_input,
        backward,
        get_logger_input,
        scheduler,
        "tot_loss",
        cur_step,
        cur_loss,
        bes_loss,
    )


def _reshape_attention(attn: Tensor) -> Tensor:
    attn = attn.transpose(-3, -2).transpose(-2, -1)
    attn = attn.reshape(attn.shape[-3], attn.shape[-2] * attn.shape[-1])
    return attn


def contrast(
    d_cfg: dict[str, dict[str, Any]],
    r_cfg: dict[str, dict[str, Any]],
    model: TokVMelNet,
    assess: TokVMelNetLoss,
    loader: DataLoader,
    test_name: str = "contrast",
):
    base_name = test_name
    for i, data_batch in enumerate(loader):
        test_name = f"{base_name}_{i}"
        # =====Model Predict=====
        tokids_lens, pad_tokids, mel_lens, pad_mel = data_batch
        model_ret, assess_ret = contrast_wrapper(
            r_cfg, model, assess, data_batch, get_model_input, get_assess_input
        )
        tokids_lens: float = tokids_lens.item()
        tok_grnd: list[str] = indexes_to_tokens(
            list(tensor2array(pad_tokids.squeeze(0))), valid_tokens, unk_token
        )
        mel_lens: float = mel_lens.item()
        pad_mel: np.ndarray = tensor2array(pad_mel.squeeze(0))
        model_ret: tuple[
            Tensor,
            Tensor,
            dict[str, dict[int, Tensor]],
            dict[str, dict[int, Tensor]],
            dict[str, Tensor],
            Tensor,
        ]
        mel_pred, mel_lens_pred, cd_attns, cd_block_outs, lr_attns, mel_bef_res = (
            model_ret
        )
        mel_pred: np.ndarray = tensor2array(mel_pred.squeeze(0))
        mel_bef_res: np.ndarray = tensor2array(mel_bef_res.squeeze(0))
        mel_lens_pred: float = mel_lens_pred.item()
        tok_attns: dict[str, np.ndarray] = {
            f"Layer-{i}": tensor2array(_reshape_attention(layer.squeeze(0)))
            for i, layer in cd_attns["encoder"].items()
        }
        mel_attns: dict[str, np.ndarray] = {
            f"Layer-{i}": tensor2array(_reshape_attention(layer.squeeze(0)))
            for i, layer in cd_attns["decoder"].items()
        }
        tok_block_outs: dict[str, np.ndarray] = {
            f"Layer-{i}": tensor2array(layer.transpose(-2, -1).squeeze(0))
            for i, layer in cd_block_outs["encoder"].items()
        }
        mel_block_outs: dict[str, np.ndarray] = {
            f"Layer-{i}": tensor2array(layer.transpose(-2, -1).squeeze(0))
            for i, layer in cd_block_outs["decoder"].items()
        }
        lr_group: dict[str, Any] = {}
        lpd_attn: Optional[Tensor] = lr_attns.get("len_predictor", None)
        if lpd_attn is not None:
            lpd_attn: dict[str, np.ndarray] = {
                f"head-{i}": tensor2array(lpd_attn[0, i, 0, :])
                for i in range(lpd_attn.shape[1])
            }
            lr_group["lpd_attn"] = lpd_attn
        stl_attn: Optional[Tensor] = lr_attns.get("seq_translator", None)
        if stl_attn is not None:
            stl_attn = tensor2array(_reshape_attention(stl_attn.squeeze(0)))
            lr_group["stl_attn"] = stl_attn
        terminal_outputs: dict[str] = {
            "tok_grnd": " ".join(
                ["".join(tok_grnd[seg[0] : seg[1]]) for seg in merge_tokens(tok_grnd)]
            ),
            "tok_len (grnd)": tokids_lens,
            "mel_len (pred, grnd)": (mel_lens_pred, mel_lens),
        }
        terminal_outputs.update(get_logger_input(assess_ret))
        # =====Dump & Display=====
        gen_dir = r_cfg["general"]["gen_dir"]
        os.makedirs(gen_dir, exist_ok=True)
        wave = mels_to_wave(mel_pred, d_cfg)
        save_wave(
            os.path.join(gen_dir, f"{test_name}.wav"),
            wave,
            d_cfg["audio"]["sample_rate"],
        )
        with open(
            os.path.join(gen_dir, f"{test_name}_model_output.json"),
            "w",
            encoding="utf-8",
        ) as wfp:
            json.dump(terminal_outputs, wfp, ensure_ascii=False, indent=2)
        for k, v in terminal_outputs.items():
            print("{}: {}".format(get_vivid_str(k, color="mgt"), v))
        if not d_cfg["mel"]["log"]:
            mel_pred = np.log(np.clip(mel_pred, a_min=d_cfg["mel"]["eps"], a_max=None))
            mel_bef_res = np.log(
                np.clip(mel_bef_res, a_min=d_cfg["mel"]["eps"], a_max=None)
            )
            pad_mel = np.log(np.clip(pad_mel, a_min=d_cfg["mel"]["eps"], a_max=None))
        display_groups: dict[str, dict[str, Any]] = {
            "model_output": {
                "mel_pred": mel_pred.T,
                "mel_bef_res": mel_bef_res.T,
                "pad_mel": pad_mel.T,
            },
            "tok_attns": tok_attns,
            "mel_attns": mel_attns,
            "tok_block_outs": tok_block_outs,
            "mel_block_outs": mel_block_outs,
            "len_regulators": lr_group,
        }
        for name, group in display_groups.items():
            if len(group):
                show_data(
                    group,
                    name,
                    save_path=os.path.join(gen_dir, f"{test_name}_{name}.png"),
                )


def generate_TTS(
    d_cfg: dict[str, dict[str, Any]],
    r_cfg: dict[str, dict[str, Any]],
    model: TokVMelNet,
    text: str,
    test_name: str = "test",
):
    terminal_outputs: dict[str, Any] = {}
    text = replace_puncts(text)  # 替换中文标点
    terminal_outputs["text"] = text
    pinyins = [
        pys[0] for pys in pypinyin.pinyin(text, style=pypinyin.Style.TONE3)
    ]  # 拼音
    tokens, merge_segs = pinyins_to_tokens(pinyins)
    token_ids: list[int] = tokens_to_indexes(
        tokens, valid_tokens, valid_tokens.index(unk_token)
    )
    tokens = indexes_to_tokens(token_ids, valid_tokens, unk_token)
    terminal_outputs["tokens"] = " ".join(
        ["".join(tokens[beg_pos:end_pos]) for beg_pos, end_pos in merge_segs]
    )
    model = model.to(r_cfg["general"]["device"])
    X_A, A_lens = to_device(
        torch.tensor(token_ids).unsqueeze(0),
        torch.tensor(len(token_ids)).view(1),
        device=r_cfg["general"]["device"],
    )
    need_attns: dict[str] = r_cfg["need_attns"]
    need_block_outs: dict[str] = r_cfg["need_block_outs"]
    tok_need_attns: list[int] = need_attns.get("tok_coder", [])
    mel_need_attns: list[int] = need_attns.get("mel_coder", [])
    t2m_lpd_need_attn: bool = need_attns.get("tok2mel_len_predictor", False)
    t2m_stl_need_attn: bool = need_attns.get("tok2mel_seq_translator", False)
    tok_need_block_outs: list[int] = need_block_outs.get("tok_coder", [])
    mel_need_block_outs: list[int] = need_block_outs.get("mel_coder", [])
    # =====Model Predict=====
    model.eval()
    with torch.no_grad():
        X_B, B_lens_pred, cd_attns, cd_block_outs, lr_attns, X_B_bef_res = model(
            X_A=X_A,
            A_lens=A_lens,
            B_lens=None,
            cd_need_attns=(tok_need_attns, mel_need_attns),
            cd_need_block_outs=(tok_need_block_outs, mel_need_block_outs),
            lr_need_attns=(t2m_lpd_need_attn, t2m_stl_need_attn),
            pos_enc_input=True,
        )
    mel_pred = tensor2array(X_B.squeeze(0))
    mel_bef_res = tensor2array(X_B_bef_res.squeeze(0))
    X_B, X_B_bef_res = None, None
    terminal_outputs["mel_len (pred)"] = B_lens_pred.item()
    display_groups: dict[str, dict[str, Any]] = {}
    for cd_name, cd_blocks in cd_attns.items():
        if len(cd_blocks):
            display_groups[f"{cd_name}-attention"] = {
                f"Layer {i}": tensor2array(_reshape_attention(block.squeeze(0)))
                for i, block in cd_blocks.items()
            }
    for cd_name, cd_blocks in cd_block_outs.items():
        if len(cd_blocks):
            display_groups[f"{cd_name}-block_out"] = {
                f"Layer {i}": tensor2array(block.squeeze(0).transpose(-2, -1))
                for i, block in cd_blocks.items()
            }
    model_out_group: dict[str, Any] = {}
    lpd_attn: Optional[Tensor] = lr_attns.get("len_predictor", None)
    if lpd_attn is not None:
        lpd_attn: dict[str, np.ndarray] = {
            f"head-{i}": tensor2array(lpd_attn[0, i, 0, :])
            for i in range(lpd_attn.shape[1])
        }
        model_out_group["len_predictor_attn"] = lpd_attn
    stl_attn: Optional[Tensor] = lr_attns.get("seq_translator", None)
    if stl_attn is not None:
        stl_attn = tensor2array(_reshape_attention(stl_attn.squeeze(0)))
        model_out_group["seq_translator_attn"] = stl_attn
    display_groups["model_output"] = model_out_group
    gen_dir = r_cfg["general"]["gen_dir"]
    # =====Dump & Display=====
    os.makedirs(gen_dir, exist_ok=True)
    wave = mels_to_wave(mel_pred, d_cfg)
    save_wave(
        os.path.join(gen_dir, f"{test_name}.wav"), wave, d_cfg["audio"]["sample_rate"]
    )
    with open(
        os.path.join(gen_dir, f"{test_name}_model_output.json"), "w", encoding="utf-8"
    ) as wfp:
        json.dump(terminal_outputs, wfp, ensure_ascii=False, indent=2)
    for k, v in terminal_outputs.items():
        print("{}: {}".format(get_vivid_str(k, color="mgt"), v))
    if not d_cfg["mel"]["log"]:
        mel_pred = np.log(np.clip(mel_pred, a_min=d_cfg["mel"]["eps"], a_max=None))
        mel_bef_res = np.log(
            np.clip(mel_bef_res, a_min=d_cfg["mel"]["eps"], a_max=None)
        )
    display_groups["model_output"].update(
        {
            "mel_pred": mel_pred.T,
            "mel_bef_res": mel_bef_res.T,
        }
    )
    for name, group in display_groups.items():
        if len(group):
            show_data(
                group,
                name,
                save_path=os.path.join(gen_dir, f"{test_name}_{name}.png"),
            )


if __name__ == "__main__":
    pass
