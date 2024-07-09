import math
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str, desc: Optional[str] = None):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.scalars: dict[str, list[float]] = {}
        self.desc: Optional[str] = desc

    def append(self, scalars: dict[str, float]):
        for name, value in scalars.items():
            if self.scalars.get(name, None) is None:
                self.scalars[name] = []
            self.scalars[name].append(value)

    def dump(self, step: Optional[int] = None, selects: Optional[list[str]] = None):
        if selects is None:
            selects = self.scalars.keys()
        for name in selects:
            values = self.scalars.get(name, [])
            if not values:
                continue
            self.writer.add_scalar(
                f"{name}/{self.desc}" if isinstance(self.desc, str) else name,
                sum(values) / len(values),
                step,
            )

    def get(self, select: str) -> Optional[float]:
        values = self.scalars.get(select, [])
        if not values:
            return None
        return sum(values) / len(values)

    def reset(self):
        self.scalars = {}

    def close(self):
        self.writer.close()
