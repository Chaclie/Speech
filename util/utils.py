from typing import Any, Optional, Union
import math, os
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
import librosa
import soundfile
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def get_vivid_str(
    origin: str, color: Optional[str] = None, style: Optional[str] = None
) -> str:
    """
    make {origin} more vivid in terminal \n
    Args:
    - color: red,grn,ylw,blu,mgt,cyn.wht
    - style: bold,itly,udln
    """
    dst = origin
    if color is not None:
        colors = ["red", "grn", "ylw", "blu", "mgt", "cyn", "wht"]
        if color in colors:
            dst = f"\033[3{colors.index(color)+1}m{dst}\033[0m"
        else:
            raise ValueError(f"color({color}) should be in {colors}")
    if style is not None:
        styles = {"bold": 1, "itly": 3, "udln": 4}
        if style in styles.keys():
            dst = f"\033[{styles[style]}m{dst}\033[0m"
        else:
            raise ValueError(f"style({style}) should be in {styles.keys()}")
    return dst


def load_wave(
    wave_path: str,
    sample_rate: int = 22050,
    trim: bool = True,
    top_db: float = 60,
    win_len: int = 256,
    hop_len: int = 1024,
    pad_lens: tuple[int, int] = (0, 0),
) -> np.ndarray:
    wave, _ = librosa.load(wave_path, sr=sample_rate)
    if trim:
        wave, _ = librosa.effects.trim(
            wave, top_db=top_db, frame_length=win_len, hop_length=hop_len
        )
    pad_len_pre, pad_len_post = max(0, pad_lens[0]), max(0, pad_lens[1])
    wave = np.concatenate([np.zeros(pad_len_pre), wave, np.zeros(pad_len_post)])
    return wave


def save_wave(wave_path: str, wave: np.ndarray, sample_rate: int) -> None:
    soundfile.write(wave_path, wave, sample_rate)


def wave_to_mels(wave: np.ndarray, d_cfg: dict[str, dict[str, Any]]) -> np.ndarray:
    """get mel spectrogram(n_frames, n_mels) from waveform"""
    mels: np.ndarray = librosa.feature.melspectrogram(
        y=wave,
        sr=d_cfg["audio"]["sample_rate"],
        n_fft=d_cfg["stft"]["n_fft"],
        hop_length=d_cfg["stft"]["hop_len"],
        win_length=d_cfg["stft"]["win_len"],
        n_mels=d_cfg["mel"]["n_mels"],
        fmin=d_cfg["mel"]["fmin"],
        fmax=d_cfg["mel"]["fmax"],
    )
    if d_cfg["mel"]["log"]:
        mels = np.log(np.clip(mels, a_min=d_cfg["mel"]["eps"], a_max=None))
    mels = mels.T
    return mels


def mels_to_wave(mels: np.ndarray, d_cfg: dict[str, dict[str, Any]]) -> np.ndarray:
    """get waveform from mel spectrogram(n_frames, n_mels)"""
    if d_cfg["mel"]["log"]:
        mels = np.exp(mels)
    mels = mels.T
    wave: np.ndarray = librosa.feature.inverse.mel_to_audio(
        mels,
        sr=d_cfg["audio"]["sample_rate"],
        n_fft=d_cfg["stft"]["n_fft"],
        hop_length=d_cfg["stft"]["hop_len"],
        win_length=d_cfg["stft"]["win_len"],
    )
    return wave


def token_to_index(token: str, tokens: list[str], default_value: int = 0) -> int:
    return tokens.index(token) if token in tokens else default_value


def index_to_token(index: int, tokens: list[str], default_value: str = "") -> str:
    return (
        tokens[index]
        if -len(tokens) <= index and index < len(tokens)
        else default_value
    )


def tokens_to_indexes(
    given: Union[list, tuple, str], tokens: list[str], default_value: int = 0
) -> Union[list, tuple, int]:
    if isinstance(given, list) or isinstance(given, tuple):
        ret = [tokens_to_indexes(symbol, tokens, default_value) for symbol in given]
        return tuple(ret) if isinstance(given, tuple) else ret
    elif isinstance(given, str):
        return token_to_index(given, tokens, default_value)


def indexes_to_tokens(
    given: Union[list, tuple, int], tokens: list[str], default_value: str = ""
) -> Union[list, tuple, str]:
    if isinstance(given, list) or isinstance(given, tuple):
        ret = [indexes_to_tokens(index, tokens, default_value) for index in given]
        return tuple(ret) if isinstance(given, tuple) else ret
    else:
        return index_to_token(given, tokens, default_value)


def to_device(*data: Any, device: str) -> Any:
    if len(data) > 1:
        return tuple(d.to(device) if isinstance(d, Tensor) else d for d in data)
    elif len(data) == 1:
        return data[0].to(device) if isinstance(data[0], Tensor) else data[0]
    return None


def get_mask_of_lengths(lengths: Tensor) -> Tensor:
    """
    get bool mask in shape[batch_size, max_length] where position(0-based) >= lengths[i] is True \n
    e.g.:
    - get lengths = [2, 4, 1]
    - return mask = [[FFTT], [FFFF], [FTTT]]
    """
    max_length = torch.max(lengths).item()
    mask = (
        torch.arange(0, max_length, device=lengths.device) >= lengths.unsqueeze(1)
    ).to(torch.bool)
    mask.requires_grad_(False)
    return mask


def get_mask_of_windows(
    length_y: int, length_x: int, win_radius: float, device: Optional[str] = None
) -> Tensor:
    """
    get bool mask in shape[length_y, length_x] where |j-i*length_y/length_x| < win_radius is True \n
    e.g.:
    - get length_y = 2, length_x = 4, win_radius = 1.5
    - return mask in shape[2, 4] = [[TTFF], [FTTT]]
    """
    mask = (
        torch.abs(
            torch.arange(length_x)
            - torch.arange(length_y).unsqueeze(1) * length_x / length_y
        )
        < win_radius
    ).to(dtype=torch.bool, device=device)
    return mask


def get_mask_of_future(length: int, device: Optional[str] = None) -> Tensor:
    """
    get bool mask in shape[length, length] where position(0-based) > i is True \n
    e.g.:
    - get length = 4
    - return mask = [[FTTT], [FFTT], [FFFT], [FFFF]]
    """
    mask = torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )
    return mask


def show_data(
    data_dic: dict[str, Union[np.ndarray, dict[str, list]]],
    title: Optional[str] = None,
    grid_shape: Optional[tuple[int, int]] = None,
    fig_size: Optional[tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    - data in 1-dim shown as curve
    - data in 2-dim shown as image
    """
    cols = int(math.ceil(math.sqrt(len(data_dic)))) or 1
    rows = int(math.ceil(len(data_dic) / cols)) or 1
    if grid_shape is not None:
        rows, cols = grid_shape
    if fig_size is None:
        fig_size = (cols * 10 / 3 + 0.5, rows * 8 / 3 + 0.5)
    fig, axs = plt.subplots(rows, cols, figsize=fig_size)
    fig.tight_layout(rect=[0.05, 0.02, 0.95, 0.95])
    if title:
        fig.suptitle(title)
    for i, (tit, data) in enumerate(data_dic.items()):
        ax: Axes = (
            axs[(i // cols, i % cols) if rows > 1 else i % cols] if cols > 1 else axs
        )
        if isinstance(data, dict):
            for name, curve in data.items():
                ax.plot(curve, label=name)
            ax.legend()
            ax.grid(True)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                ax.plot(data)
                ax.grid(True)
            elif data.ndim == 2:
                fig.colorbar(ax.imshow(data, aspect="auto", origin="lower"))
            else:
                raise ValueError(
                    f"shape of data({data.shape}) should be in 1 or 2 dims"
                )
        else:
            raise TypeError(
                f"type of data({type(data)}) should be Union[np.ndarray, dict[str, list]]"
            )
        ax.set_title(tit)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def load_ckpt(
    ckpt_path: str, show_hint: bool = True
) -> tuple[str, nn.Module, Optimizer, int, float, float]:
    """hint, model, optim, cur_step, cur_loss, bes_loss"""
    ckpt = torch.load(ckpt_path)
    model = ckpt["model"]
    optim = ckpt["optim"]
    cur_step = ckpt["step"]
    cur_loss = ckpt["cur_loss"]
    bes_loss = ckpt["bes_loss"]
    load_hint = "{} ckpt at step {:08d} with {} from {}".format(
        get_vivid_str("Load", color="grn"),
        cur_step,
        (
            "loss={:.6f}(best={:.6f})".format(cur_loss, bes_loss)
            if bes_loss < cur_loss
            else "{} loss={:.6f}".format(get_vivid_str("best", color="grn"), cur_loss)
        ),
        ckpt_path,
    )
    if show_hint:
        print(load_hint)
    return load_hint, model, optim, cur_step + 1, cur_loss, bes_loss


def save_ckpt(
    ckpt_path: str,
    model: nn.Module,
    optim: Optimizer,
    cur_step: int,
    cur_loss: float,
    bes_loss: float,
    show_hint: bool = True,
) -> str:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model": model,
            "optim": optim,
            "step": cur_step,
            "cur_loss": cur_loss,
            "bes_loss": bes_loss,
        },
        ckpt_path,
    )
    save_hint = "{} ckpt at step {:08d} with {} from {}".format(
        get_vivid_str("Save", color="grn"),
        cur_step,
        (
            "loss={:.6f}(best={:.6f})".format(cur_loss, bes_loss)
            if bes_loss < cur_loss
            else "{} loss={:.6f}".format(get_vivid_str("best", color="grn"), cur_loss)
        ),
        ckpt_path,
    )
    if show_hint:
        print(save_hint)
    return save_hint


def get_lr(optim: Optimizer) -> float:
    return optim.param_groups[0]["lr"]


def set_lr(optim: Optimizer, lr: float):
    for param_group in optim.param_groups:
        param_group["lr"] = lr


def tensor2array(ts: Tensor) -> np.ndarray:
    return ts.detach().cpu().numpy()
