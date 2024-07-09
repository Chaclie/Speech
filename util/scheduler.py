from typing import Any


class Scheduler:
    def __init__(self, r_cfg: dict[str, dict[str, Any]]):
        self.lr_init = r_cfg["optim"]["lr_init"]
        self.warmup_steps = r_cfg["scheduler"]["warmup_steps"]

    def get_lr(self, cur_step: int, cur_lr: float) -> float:
        return self.lr_init * min(cur_step**-0.5, cur_step * self.warmup_steps**-1.5)
