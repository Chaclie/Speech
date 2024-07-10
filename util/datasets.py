import os, json, random
from typing import Any, Optional, Union
from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

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


class TokMelDataset(Dataset):
    def __init__(
        self,
        meta_infos: list[tuple[str, list[int]]],
        mel_dir: str,
        load_from_disk: bool = True,
    ):
        super(TokMelDataset, self).__init__()
        self.dataset: list[tuple[str, list[int]]] = meta_infos
        self.mel_dir: str = mel_dir
        self.load_from_disk: bool = load_from_disk
        if not load_from_disk:
            self.mel_list: list[Tensor] = [
                torch.from_numpy(np.load(os.path.join(mel_dir, "{}.npy".format(name))))
                for name, _ in tqdm(
                    self.dataset,
                    desc="Load Mels",
                    ascii=True,
                    bar_format="{desc}: {percentage:3.1f}%={n_fmt}/{total_fmt}[{remaining}{postfix}]",
                )
            ]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[list[int], Tensor]:
        name, tokids = self.dataset[idx]
        if self.load_from_disk:
            mel = torch.from_numpy(np.load(os.path.join(self.mel_dir, f"{name}.npy")))
        else:
            mel = self.mel_list[idx]
        return tokids, mel


class TokMelCollate:
    def __init__(self):
        pass

    def __call__(
        self, batch: list[tuple[list[int], Tensor]]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = len(batch)
        # pad tokids in the order of descending length
        tokids_lens, orders = torch.tensor([len(tokids) for tokids, _ in batch]).sort(
            dim=0, descending=True
        )
        max_tokids_len = tokids_lens[0]
        pad_tokids = torch.zeros([batch_size, max_tokids_len], dtype=torch.int)
        for i in range(batch_size):
            pad_tokids[i, : len(batch[orders[i]][0])] = torch.tensor(
                batch[orders[i]][0]
            )
        # pad mels in the same order
        mel_dim = batch[0][1].shape[1]
        mel_lens = torch.tensor(
            [batch[orders[i]][1].shape[0] for i in range(batch_size)]
        )
        max_mel_len = torch.max(mel_lens)
        pad_mel = torch.zeros([batch_size, max_mel_len, mel_dim])
        pad_stp = torch.zeros([batch_size, max_mel_len])
        for i in range(batch_size):
            pad_mel[i, : mel_lens[i], :] = batch[orders[i]][1]
            pad_stp[i, : mel_lens[i] - 1] = torch.exp(
                torch.linspace(-20, -1, mel_lens[i] - 1)
            )
            pad_stp[i, mel_lens[i] - 1 :] = 1.0
        # tokids_lens/mel_lens: [batch_size]
        # pad_tokids: [batch_size, max_tokids_len]
        # pad_mel: [batch_size, max_mel_len, mel_dim]
        # pad_stp: [batch_size, max_mel_len]
        return tokids_lens, pad_tokids, mel_lens, pad_mel, pad_stp
