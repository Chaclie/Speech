import torch
from typing import Any

from .util.textparser import valid_tokens, _pad as pad_token

data_process_config: dict[str, dict[str, Any]] = {
    "self": {"name": "data_process_config"},
    "audio": {
        "sample_rate": 22050,
        "trim": True,
        "top_db": 20,
        "pad_lens": (0, 0),  # (int(22050 * 0.05), int(22050 * 0.1)),
    },
    "stft": {"n_fft": 1024, "hop_len": 256, "win_len": 1024},
    "mel": {"n_mels": 80, "fmin": 0.0, "fmax": 8000.0, "log": True, "eps": 1e-5},
}

transformerTTS_model_config: dict[str, dict[str, Any]] = {
    "self": {"name": "transformerTTS_model_config"},
    "tok_prenet": {
        "max_tok_len": 100,
        "n_toks": len(valid_tokens),
        "embed_dim": 512,
        "pad_idx": valid_tokens.index(pad_token),
        "conv_n_layers": 3,
        "conv_k_size": 5,
        "conv_hid_dim": 512,
    },
    "mel_prenet": {
        "max_mel_len": 1024,
        "n_mels": data_process_config["mel"]["n_mels"],
        "n_layers": 2,
        "hid_dim": 256,
    },
    "encoder": {
        "n_blocks": 3,
        "att_n_heads": 4,
        "att_in_dim_q": 512,
        "att_hid_dim": 128,
        "att_out_dim_v": 128,
        "ffn_hid_dim": 2048,
        "norm_first": False,
    },
    "decoder": {
        "n_blocks": 3,
        "att_n_heads": 4,
        "att_in_dim_q": 512,
        "att_hid_dim": 128,
        "att_out_dim_v": 128,
        "ffn_hid_dim": 2048,
        "norm_first": False,
    },
    "mel_postnet": {
        "conv_n_layers": 5,
        "conv_k_size": 5,
        "conv_hid_dim": 512,
    },
    "loss": {},
}
transformerTTS_runtime_config: dict[str, dict[str, Any]] = {
    "self": {"name": "transformerTTS_runtime_config"},
    "general": {
        "seed": 11451419,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # path relative to the whole module
        "dat_dir": "Runtime/data/biaobei_trim_nopad",  #
        "log_dir": "Runtime/log/transformerTTS/exp1",
        "gen_dir": "Runtime/gen/transformerTTS/exp1",
        "sav_dir": "Runtime/cache/transformerTTS/exp1",
        "total_size": 10000,
        "train_size": 9800,
        "valid_size": 200,
    },
    "need_attns": {
        "enc_self": [],
        "dec_self": [],
        "dec_cros": [],
    },
    "need_block_outs": {
        "encoder": [],
        "decoder": [],
    },
    "dropout": {
        "tok_prenet": 0.5,
        "mel_prenet": 0.5,
        "encoder": 0.1,
        "decoder": 0.1,
        "mel_postnet": 0.5,
    },
    "optim": {
        "batch_size": 8,
        "lr_init": transformerTTS_model_config["encoder"]["att_in_dim_q"] ** -0.5,
        "betas": [0.9, 0.98],
        "eps": 1e-9,
        "weight_decay": 0,  # 1e-8,  # L2正则化
        "grad_clip_threshold": 1.0,
    },
    "scheduler": {"warmup_steps": 4000},
    "train": {
        "mask_output": True,
        # a batch = a step
        "max_step": 9800 * 50 / 8 + 2,
        "train_step": 128 / 8,
        "valid_step": 9800 / 5 / 8,
        "save_step": 9800 / 8,
    },
    "loader": {"load_from_disk": False},
}

TokVMelNet_model_config: dict[str, dict[str, Any]] = {
    "self": {"name": "TokVMelNet_model_config"},
    "tok_coder": {
        "n_blocks": 4,
        "att_n_heads": 4,
        "att_in_dim_q": 512,
        "att_hid_dim": 128,
        "att_out_dim_v": 128,
        "conv_hid_dim": 2048,
        "conv_k_size": 5,
        "norm_first": False,
    },
    "mel_coder": {
        "n_blocks": 4,
        "att_n_heads": 4,
        "att_in_dim_q": 512,
        "att_hid_dim": 128,
        "att_out_dim_v": 128,
        "conv_hid_dim": 2048,
        "conv_k_size": 9,
        "norm_first": False,
    },
    "tok2mel_lpd": {
        "att_n_heads": 4,
        "att_hid_dim": 32,
        "att_out_dim_v": 8,
        "conv_hid_dim": 32,  # 模型结构被修改而弃用
        "conv_k_size": 1,  # 模型结构被修改而弃用
        "norm_first": False,  # 模型结构被修改而弃用
    },
    "tok2mel_stl": {
        "att_n_heads": 4,
        "att_hid_dim": 128,
        "att_out_dim_v": 128,
        "conv_hid_dim": 2048,
        "conv_k_size": 9,
        "norm_first": False,
    },
    "tok_prenet": {
        "max_tok_len": 128,
        "n_toks": len(valid_tokens),
        "embed_dim": 512,
        "pad_idx": valid_tokens.index(pad_token),
        "conv_n_layers": 3,
        "conv_k_size": 5,
        "conv_hid_dim": 512,
        "proj_first": False,
    },
    "mel_postnet": {
        "max_mel_len": 1024,
        "n_mels": data_process_config["mel"]["n_mels"],
        "conv_n_layers": 4,
        "conv_k_size": 7,
        "conv_hid_dim": 512,
        "proj_first": True,
    },
    "loss": {},
}
TokVMelNet_runtime_config: dict[str, dict[str, Any]] = {
    "self": {"name": "TokVMelNet_runtime_config"},
    "general": {
        "seed": 11451419,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # path relative to the whole module
        "dat_dir": "Runtime/data/biaobei_trim_nopad",  #
        "log_dir": "Runtime/log/TokVMelNet/exp1",
        "gen_dir": "Runtime/gen/TokVMelNet/exp1",
        "sav_dir": "Runtime/cache/TokVMelNet/exp1",
        "total_size": 10000,
        "train_size": 9800,
        "valid_size": 200,
    },
    "need_attns": {
        "tok_coder": [],
        "tok2mel_len_predictor": False,
        "tok2mel_seq_translator": False,
        "mel_coder": [],
    },
    "need_block_outs": {
        "tok_coder": [],
        "mel_coder": [],
    },
    "dropout": {
        "tok_prenet": 0.5,
        "tok_coder": 0.1,
        "tok2mel_len_predictor": 0.5,
        "tok2mel_seq_translator": 0.5,
        "mel_coder": 0.1,
        "mel_postnet": 0.5,
    },
    "optim": {
        "batch_size": 16,
        "lr_init": TokVMelNet_model_config["tok_coder"]["att_in_dim_q"] ** -0.5,
        "betas": [0.9, 0.98],
        "eps": 1e-9,
        "weight_decay": 0,  # 1e-8,  # L2正则化
        "grad_clip_threshold": 1.0,
    },
    "scheduler": {"warmup_steps": 4000},
    "train": {
        "mask_output": True,
        # a batch = a step
        "max_step": 9800 * 5 / 16 + 2,
        "train_step": 128 / 16,
        "valid_step": 9800 / 5 / 16,
        "save_step": 9800 * 2 / 16,
    },
    "loader": {"load_from_disk": False},
}
