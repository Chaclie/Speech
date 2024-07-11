import os
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
    load_ckpt,
    show_data,
    tensor2array,
)
from ..util.textparser import (
    valid_tokens,
    _unk as unk_token,
    replace_puncts,
    pinyins_to_tokens,
)
from ..util.datasets import TokMelDataset, TokMelCollate
from ..util.scheduler import Scheduler
from ..util.runhelper import train_wrapper, contrast_wrapper
from ..model.transformerTTS import (
    TransformerTTS,
    TransformerTTSLoss,
    get_model_input,
    get_assess_input,
    backward,
    get_logger_input,
)


def train(
    r_cfg: dict[str, dict[str, Any]],
    model: TransformerTTS,
    assess: TransformerTTSLoss,
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


def contrast(
    d_cfg: dict[str, dict[str, Any]],
    r_cfg: dict[str, dict[str, Any]],
    model: TransformerTTS,
    assess: TransformerTTSLoss,
    loader: DataLoader,
    test_name: str = "contrast",
):
    base_name = test_name
    for i, data_batch in enumerate(loader):
        test_name = f"{base_name}_{i}"
        tokids_lens, pad_tokids, mel_lens, pad_mel, pad_stp = data_batch
        model_ret, assess_ret = contrast_wrapper(
            r_cfg, model, assess, data_batch, get_model_input, get_assess_input
        )
        model_ret: tuple[Tensor, Tensor, Tensor, dict[str, list[Optional[Tensor]]]]
        assess_ret: tuple[Tensor, Tensor, Tensor]
        mel_out, mel_out_post, stp_out, attns, block_outs = model_ret
        mel_out: np.ndarray = tensor2array(mel_out.squeeze(0))
        mel_out_post: np.ndarray = tensor2array(mel_out_post.squeeze(0))
        pad_mel: np.ndarray = tensor2array(pad_mel.squeeze(0))
        stp_out: np.ndarray = tensor2array(stp_out.squeeze(0))
        pad_stp: np.ndarray = tensor2array(pad_stp.squeeze(0))
        attns: dict[str, dict[str, np.ndarray]] = {
            name: {
                f"Layer {i}": block.transpose(-3, -2)
                .transpose(-2, -1)
                .reshape(
                    *block.shape[:-3],
                    block.shape[-2],
                    block.shape[-3] * block.shape[-1],
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                for i, block in blocks.items()
            }
            for name, blocks in attns.items()
        }
        block_outs: dict[str, dict[str, np.ndarray]] = {
            name: {
                f"Layer {i}": tensor2array(block.transpose(-2, -1).squeeze(0))
                for i, block in blocks.items()
            }
            for name, blocks in block_outs.items()
        }
        gen_dir = r_cfg["general"]["gen_dir"]
        os.makedirs(gen_dir, exist_ok=True)
        wave = mels_to_wave(mel_out_post, d_cfg)
        save_wave(
            os.path.join(gen_dir, f"{test_name}.wav"),
            wave,
            d_cfg["audio"]["sample_rate"],
        )
        if not d_cfg["mel"]["log"]:
            mel_out = np.log(np.clip(mel_out, a_min=d_cfg["mel"]["eps"], a_max=None))
            mel_out_post = np.log(
                np.clip(mel_out_post, a_min=d_cfg["mel"]["eps"], a_max=None)
            )
            pad_mel = np.log(np.clip(pad_mel, a_min=d_cfg["mel"]["eps"], a_max=None))
        show_data(
            {
                "mel_out": mel_out.T,
                "mel_out_post": mel_out_post.T,
                "pad_mel": pad_mel.T,
                "stp_out": stp_out,
                "pad_stp": pad_stp,
            },
            "model_output",
            save_path=os.path.join(gen_dir, f"{test_name}_model_output.png"),
        )
        for name, blocks in attns.items():
            if len(blocks):
                show_data(
                    blocks,
                    name,
                    save_path=os.path.join(gen_dir, f"{test_name}_{name}.png"),
                )
        for name, blocks in block_outs.items():
            if len(blocks):
                show_data(
                    blocks,
                    name,
                    save_path=os.path.join(gen_dir, f"{test_name}_{name}_out.png"),
                )


def generate(
    d_cfg: dict[str, dict[str, Any]],
    r_cfg: dict[str, dict[str, Any]],
    model: TransformerTTS,
    text: str,
    test_name: str = "test",
):
    """
    Args:
    - d_cfg:
        - audio: sample_rate
        - stft: n_fft, hop_length, win_length
    - r_cfg:
        - general: device, gen_dir
        - need_attns: enc_self, dec_self, dec_cros
    """
    text = replace_puncts(text)  # 替换中文标点
    print("{}: {}".format(get_vivid_str("text", color="mgt"), text))
    pinyins = [
        pys[0] for pys in pypinyin.pinyin(text, style=pypinyin.Style.TONE3)
    ]  # 拼音
    tokens, merge_segs = pinyins_to_tokens(pinyins)
    token_ids = tokens_to_indexes(tokens, valid_tokens, valid_tokens.index(unk_token))
    tokens = indexes_to_tokens(token_ids, valid_tokens, unk_token)
    print(
        "{}: {}".format(
            get_vivid_str("tokens", color="mgt"),
            " ".join(
                ["".join(tokens[beg_pos:end_pos]) for beg_pos, end_pos in merge_segs]
            ),
        )
    )
    model = model.to(r_cfg["general"]["device"])
    token_ids_tensor = to_device(
        torch.tensor(token_ids), device=r_cfg["general"]["device"]
    )
    need_attns: dict[str, list[int]] = {
        name: r_cfg["need_attns"].get(name, [])
        for name in ["enc_self", "dec_self", "dec_cros"]
    }
    need_block_outs: dict[str, list[int]] = {
        name: r_cfg["need_block_outs"].get(name, []) for name in ["encoder", "decoder"]
    }
    model.eval()
    with torch.no_grad():
        mel_out, mel_out_post, stp_out, attns, block_outs = model.synthesize(
            token_ids_tensor, need_attns=need_attns, need_block_outs=need_block_outs
        )
    mel_out: np.ndarray = tensor2array(mel_out)
    mel_out_post: np.ndarray = tensor2array(mel_out_post)
    stp_out: np.ndarray = tensor2array(stp_out)
    attns: dict[str, dict[str, np.ndarray]] = {
        name: {
            f"Layer {i}": block.transpose(-3, -2)
            .transpose(-2, -1)
            .reshape(
                *block.shape[:-3], block.shape[-2], block.shape[-3] * block.shape[-1]
            )
            .detach()
            .cpu()
            .numpy()
            for i, block in blocks.items()
        }
        for name, blocks in attns.items()
    }
    block_outs: dict[str, dict[str, np.ndarray]] = {
        name: {
            f"Layer {i}": tensor2array(block.transpose(-2, -1))
            for i, block in blocks.items()
        }
        for name, blocks in block_outs.items()
    }
    gen_dir = r_cfg["general"]["gen_dir"]
    os.makedirs(gen_dir, exist_ok=True)
    wave = mels_to_wave(mel_out_post, d_cfg)
    save_wave(
        os.path.join(gen_dir, f"{test_name}.wav"), wave, d_cfg["audio"]["sample_rate"]
    )
    if not d_cfg["mel"]["log"]:
        mel_out = np.log(np.clip(mel_out, a_min=d_cfg["mel"]["eps"], a_max=None))
        mel_out_post = np.log(
            np.clip(mel_out_post, a_min=d_cfg["mel"]["eps"], a_max=None)
        )
    show_data(
        {"mel_out": mel_out.T, "mel_out_post": mel_out_post.T, "stp_out": stp_out},
        "model_output",
        save_path=os.path.join(gen_dir, f"{test_name}_model_output.png"),
    )
    print(
        "dec_self attns: {}".format(
            [len(np.nonzero(block)[0]) for i, block in attns["dec_self"].items()]
        )
    )
    for name, blocks in attns.items():
        if len(blocks):
            show_data(
                blocks, name, save_path=os.path.join(gen_dir, f"{test_name}_{name}.png")
            )
    for name, blocks in block_outs.items():
        if len(blocks):
            show_data(
                blocks,
                name,
                save_path=os.path.join(gen_dir, f"{test_name}_{name}_out.png"),
            )
