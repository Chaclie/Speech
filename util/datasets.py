import os, json, random
from typing import Any, Optional, Union
from tqdm import tqdm
import numpy as np

from .utils import load_wave, wave_to_mels


def build_from_biaobei(
    in_dir: str, out_dir: str, d_cfg: dict[str, dict[str, Any]]
) -> None:
    if not os.path.isdir(in_dir):
        raise ValueError(f"there is no in_dir={in_dir}")
    meta_file = os.path.join(in_dir, "ProsodyLabeling", "000001-010000.txt")
    wave_dir = os.path.join(in_dir, "Wave")
    text_file = os.path.join(out_dir, "text.txt")
    dcfg_file = os.path.join(out_dir, "{}.json".format(d_cfg["self"]["name"]))
    mels_dir = os.path.join(out_dir, "mels")
    os.makedirs(mels_dir, exist_ok=True)
    with open(meta_file, "r", encoding="utf-8") as rfp, open(
        text_file, "w", encoding="utf-8"
    ) as wfp:
        for i in tqdm(
            range(0, 10000),
            desc="Build from Biaobei",
            ascii=True,
            bar_format="{desc}: {percentage:3.1f}%={n_fmt}/{total_fmt}[{remaining}{postfix}]",
        ):
            chinese = rfp.readline().strip().split()
            base_name, chinese = chinese[0], chinese[1]
            pinyin = rfp.readline().strip()
            wfp.write(f"{base_name}|{pinyin}\n")
            np.save(
                os.path.join(mels_dir, f"{base_name}.npy"),
                wave_to_mels(
                    load_wave(
                        os.path.join(wave_dir, f"{base_name}.wav"),
                        d_cfg["audio"]["sample_rate"],
                        d_cfg["audio"]["trim"],
                        d_cfg["audio"]["top_db"],
                        d_cfg["stft"]["win_len"],
                        d_cfg["stft"]["hop_len"],
                        d_cfg["audio"]["pad_lens"],
                    ),
                    d_cfg,
                ),
            )
    with open(dcfg_file, "w", encoding="utf-8") as wfp:
        json.dump(d_cfg, wfp, ensure_ascii=False, indent=2)


def load_dataset(in_dir: str) -> list[tuple[str, str]]:
    if not os.path.isdir(in_dir):
        raise ValueError(f"there is no in_dir={in_dir}")
    text_file = os.path.join(in_dir, "text.txt")
    meta_infos: list[tuple[str, str]] = []
    with open(text_file, "r", encoding="utf-8") as rfp:
        for line in rfp:
            if not line:
                break
            meta_infos.append(tuple(line.strip().split("|")))
    return meta_infos


def split_dataset(
    total: int,
    train: Union[int, float] = 1.0,
    valid: Union[int, float] = 0.0,
    seed: Optional[int] = None,
) -> tuple[list[int], list[int]]:
    indexes = list(range(total))
    if seed is not None:
        random.seed(seed)
        random.shuffle(indexes)
    if isinstance(train, float):
        train = int(train * total)
    if isinstance(valid, float):
        valid = int(valid * total)
    if train + valid > total:
        raise ValueError(
            f"train_num({train}) + valid_num({valid}) > total_num({total})"
        )
    train_indexes, valid_indexes = indexes[:train], indexes[train : train + valid]
    return train_indexes, valid_indexes
