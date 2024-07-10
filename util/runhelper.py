import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Optional, Callable

from .scheduler import Scheduler
from .logger import Logger
from .utils import to_device, get_lr, set_lr, save_ckpt


def train_wrapper(
    r_cfg: dict[str, dict[str, Any]],
    model: nn.Module,
    assess: nn.Module,
    optim: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    get_model_input: Callable[[Any, dict[str, dict[str, Any]]], Any],
    get_assess_input: Callable[[Any, Any], Any],
    backward: Callable[[Any], None],
    get_logger_input: Callable[[Any], dict[str, float]],
    scheduler: Optional[Scheduler] = None,
    unique_assess_name: str = "loss",
    cur_step: int = 1,
    cur_loss: float = float("inf"),
    bes_loss: float = float("inf"),
):
    """
    Args:
    - <r_cfg>:
        - general: device, log_dir, sav_dir
        - optim: grad_clip_threshold
        - train: max_step, train_step, valid_step, save_step
    - <assess> is loss calculator of <model>
    - get_model_input(batch, r_cfg) -> model_input
    - get_assess_input(batch, model_output) -> assess_input ( model_output=model(model_input) )
    - backward(assess_output) ( assess_output=assess(assess_input) )
    - get_logger_input(assess_output) -> logger_input
    - <scheduler> schedules lr of <optim>
    - unique_assess_name: the name of the assessment used for comparison of <cur_loss> and <bes_loss>
    """
    device = r_cfg["general"]["device"]
    log_dir = r_cfg["general"]["log_dir"]
    sav_dir = r_cfg["general"]["sav_dir"]
    grad_clip_threshold = r_cfg["optim"]["grad_clip_threshold"]
    max_step = r_cfg["train"]["max_step"]
    train_step = r_cfg["train"]["train_step"]
    valid_step = r_cfg["train"]["valid_step"]
    save_step = r_cfg["train"]["save_step"]
    step_bar = tqdm(
        total=max_step,
        desc="Step",
        position=0,
        ascii=True,
        bar_format="{desc}: {percentage:3.1f}%={n_fmt}/{total_fmt}[{remaining}{postfix}]",
    )
    step_bar.update(cur_step)
    train_epoch_bar = tqdm(
        total=len(train_loader),
        desc="Train Epoch",
        position=1,
        ascii=True,
        bar_format="{desc}: {percentage:3.1f}%={n_fmt}/{total_fmt}[{remaining}{postfix}]",
    )
    valid_epoch_bar = tqdm(
        total=len(valid_loader),
        desc="Valid Epoch",
        position=2,
        ascii=True,
        bar_format="{desc}: {percentage:3.1f}%={n_fmt}/{total_fmt}[{remaining}{postfix}]",
    )
    save_info = tqdm(
        total=1, desc="Save Info", position=3, bar_format="{desc}{postfix}"
    )
    train_logger, valid_logger = Logger(log_dir, "train"), Logger(log_dir, "valid")
    model, assess = model.to(device), assess.to(device)
    model.train(), assess.train()
    model.zero_grad()
    while True:
        train_epoch_bar.reset()
        for train_batch in train_loader:
            train_batch = to_device(*train_batch, device=device)
            model_ret = get_model_input(train_batch, r_cfg)
            model_ret = (
                model(*model_ret) if isinstance(model_ret, tuple) else model(model_ret)
            )
            assess_ret = get_assess_input(train_batch, model_ret)
            assess_ret = (
                assess(*assess_ret)
                if isinstance(assess_ret, tuple)
                else assess(assess_ret)
            )
            backward(assess_ret)
            train_logger.append(get_logger_input(assess_ret))
            if cur_step % train_step == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)
                cur_lr = get_lr(optim)
                if scheduler is not None:
                    cur_lr = scheduler.get_lr(cur_step, cur_lr)
                    set_lr(optim, cur_lr)
                train_logger.append({"lr": cur_lr})
                optim.step()
                optim.zero_grad()
                train_logger.dump(cur_step)
                train_logger.reset()
            train_epoch_bar.update(1)
            if cur_step % valid_step == 0:
                valid_epoch_bar.reset()
                model.eval(), assess.eval()
                with torch.no_grad():
                    for valid_batch in valid_loader:
                        valid_batch = to_device(*valid_batch, device=device)
                        model_ret = get_model_input(valid_batch, r_cfg)
                        model_ret = (
                            model(*model_ret)
                            if isinstance(model_ret, tuple)
                            else model(model_ret)
                        )
                        assess_ret = get_assess_input(valid_batch, model_ret)
                        assess_ret = (
                            assess(*assess_ret)
                            if isinstance(assess_ret, tuple)
                            else assess(assess_ret)
                        )
                        valid_logger.append(get_logger_input(assess_ret))
                        valid_epoch_bar.update(1)
                cur_loss = valid_logger.get(unique_assess_name)
                if cur_loss is not None and bes_loss > cur_loss:
                    bes_loss = cur_loss
                    save_info.set_postfix_str(
                        save_ckpt(
                            os.path.join(sav_dir, "ckpt_best.pth"),
                            model,
                            optim,
                            cur_step,
                            cur_loss,
                            bes_loss,
                            False,
                        )
                    )
                valid_logger.dump(cur_step)
                valid_logger.reset()
                model.train(), assess.train()
            if cur_step % save_step == 0:
                save_info.set_postfix_str(
                    save_ckpt(
                        os.path.join(sav_dir, f"ckpt_{cur_step:08d}.pth"),
                        model,
                        optim,
                        cur_step,
                        cur_loss,
                        bes_loss,
                        False,
                    )
                )
            cur_step += 1
            step_bar.update(1)
            if cur_step >= max_step:
                train_logger.close(), valid_logger.close()
                return


def contrast_wrapper(
    r_cfg: dict[str, dict[str, Any]],
    model: nn.Module,
    assess: nn.Module,
    data_batch: Any,
    get_model_input: Callable[[Any, dict[str, dict[str, Any]]], Any],
    get_assess_input: Callable[[Any, Any], Any],
) -> tuple[Any, Any]:
    device = r_cfg["general"]["device"]
    model, assess = model.to(device), assess.to(device)
    model.eval(), assess.eval()
    with torch.no_grad():
        data_batch = to_device(*data_batch, device=device)
        model_ret = get_model_input(data_batch, r_cfg)
        model_ret = (
            model(*model_ret) if isinstance(model_ret, tuple) else model(model_ret)
        )
        assess_ret = get_assess_input(data_batch, model_ret)
        assess_ret = (
            assess(*assess_ret) if isinstance(assess_ret, tuple) else assess(assess_ret)
        )
    return model_ret, assess_ret
