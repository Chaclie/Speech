import os, random, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from .util.textparser import pinyins_to_tokens, valid_tokens, _unk as unk_token
from .util.utils import load_ckpt, tokens_to_indexes, show_data
from .util.datasets import (
    TokMelDataset,
    build_from_biaobei,
    load_dataset,
    split_dataset,
)
from .util.scheduler import Scheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument("--gen", default="")
    parser.add_argument("--contrast", default="")
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    model_name: str = args.model
    gen_input: str = args.gen
    contrast_input: str = args.contrast
    test_name: str = args.name
    gen_mode: bool = len(gen_input) > 0
    contrast_mode: bool = len(contrast_input) > 0
    test_name: str = test_name if test_name else ("test" if gen_mode else "contrast")
    contrast_input: list[str] = contrast_input.split()
    valid_models: list[str] = ["transformerTTS"]
    if model_name == valid_models[0]:
        from .config import (
            data_process_config as d_cfg,
            transformerTTS_model_config as m_cfg,
            transformerTTS_runtime_config as r_cfg,
        )
        from .util.datasets import TokMelCollate as Collate
        from .model.transformerTTS import (
            TransformerTTS as Model,
            TransformerTTSLoss as Loss,
        )
        from .run.transformerTTS import train, contrast, generate as generate_TTS
    else:
        raise ValueError(f"model_name({model_name}) should be in {valid_models}")
    # -----Init Config-----
    if gen_mode or contrast_mode:
        if model_name == valid_models[0]:
            r_cfg["need_attns"] = {
                "enc_self": [0, 1, 2, 3, 4, 5],
                "dec_self": [0, 1, 2, 3, 4, 5],
                "dec_cros": [0, 1, 2, 3, 4, 5],
            }
            r_cfg["need_block_outs"] = {
                "encoder": [0, 1, 2, 3, 4, 5],
                "decoder": [0, 1, 2, 3, 4, 5],
            }
    sav_dir = r_cfg["general"]["gen_dir" if gen_mode else "sav_dir"]
    os.makedirs(sav_dir, exist_ok=True)
    m_cfg_path = os.path.join(sav_dir, "{}.json".format(m_cfg["self"]["name"]))
    r_cfg_path = os.path.join(sav_dir, "{}.json".format(r_cfg["self"]["name"]))
    with open(m_cfg_path, "w", encoding="utf-8") as wfp:
        json.dump(m_cfg, wfp, ensure_ascii=False, indent=2)
    with open(r_cfg_path, "w", encoding="utf-8") as wfp:
        json.dump(r_cfg, wfp, ensure_ascii=False, indent=2)
    # -----Init Random-----
    seed = r_cfg["general"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # -----Create Model-----
    ckpt_path = f"Runtime/cache/{model_name}/exp1/ckpt_best.pth"
    model = Model(m_cfg)
    optim = Adam(
        model.parameters(),
        lr=r_cfg["optim"]["lr_init"],
        betas=r_cfg["optim"]["betas"],
        eps=r_cfg["optim"]["eps"],
        weight_decay=r_cfg["optim"]["weight_decay"],
    )
    if os.path.isfile(ckpt_path):
        _, cur_step, cur_loss, bes_loss = load_ckpt(ckpt_path, model, optim, True)
    else:
        cur_step, cur_loss, bes_loss = 1, float("inf"), float("inf")
    assess = Loss(m_cfg)
    scheduler = Scheduler(r_cfg)
    if model_name == valid_models[0]:
        print(
            "tok_alpha:{} mel_alpha:{}".format(
                model.get_parameter("tok_pos_alpha").item(),
                model.get_parameter("mel_pos_alpha").item(),
            )
        )
    if gen_mode:
        generate_TTS(d_cfg, r_cfg, model, gen_input, test_name)
    elif contrast_mode:
        dat_dir = r_cfg["general"]["dat_dir"]
        meta_infos = load_dataset(dat_dir)
        processed_meta_infos = [
            (
                info[0],
                tokens_to_indexes(
                    pinyins_to_tokens(info[1].split())[0],
                    valid_tokens,
                    valid_tokens.index(unk_token),
                ),
            )
            for info in meta_infos
        ]
        contrast_set = TokMelDataset(
            [pmi for pmi in processed_meta_infos if pmi[0] in contrast_input],
            os.path.join(dat_dir, "mels"),
            r_cfg["loader"]["load_from_disk"],
        )
        contrast_loader = DataLoader(contrast_set, 1, False, collate_fn=Collate())
        contrast(d_cfg, r_cfg, model, assess, contrast_loader, test_name)
    else:
        # -----Load Dataset-----
        # path relative to the whole module
        biaobei_original_dir = "../data/data-baker"
        dat_dir = r_cfg["general"]["dat_dir"]
        if not os.path.isdir(dat_dir):
            build_from_biaobei(biaobei_original_dir, dat_dir, d_cfg)
        meta_infos = load_dataset(dat_dir)
        processed_meta_infos = [
            (
                info[0],
                tokens_to_indexes(
                    pinyins_to_tokens(info[1].split())[0],
                    valid_tokens,
                    valid_tokens.index(unk_token),
                ),
            )
            for info in meta_infos
        ]
        train_indexes, valid_indexes = split_dataset(
            r_cfg["general"]["total_size"],
            r_cfg["general"]["train_size"],
            r_cfg["general"]["valid_size"],
            seed,
        )
        train_set = TokMelDataset(
            [processed_meta_infos[i] for i in train_indexes],
            os.path.join(dat_dir, "mels"),
            r_cfg["loader"]["load_from_disk"],
        )
        valid_set = TokMelDataset(
            [processed_meta_infos[i] for i in valid_indexes],
            os.path.join(dat_dir, "mels"),
            r_cfg["loader"]["load_from_disk"],
        )
        train_loader = DataLoader(
            train_set, r_cfg["optim"]["batch_size"], True, collate_fn=Collate()
        )
        valid_loader = DataLoader(
            valid_set, r_cfg["optim"]["batch_size"], False, collate_fn=Collate()
        )
        train(
            r_cfg,
            model,
            assess,
            optim,
            train_loader,
            valid_loader,
            scheduler,
            cur_step,
            cur_loss,
            bes_loss,
        )
