import random
from typing import Optional

import numpy as np
import torch
from torch.optim import Adam, AdamW, SGD


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(
    optimizer_name: str,
    parameters,
    learning_rate: float,
    weight_decay: float,
):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return SGD(parameters, lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer type: {optimizer_name}")


def get_gpu_memory(device: Optional[torch.device] = None) -> float:
    if device is None:
        if not torch.cuda.is_available():
            return 0.0
        device = torch.device("cuda")

    if not torch.cuda.is_available():
        return 0.0

    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    return float(allocated) / 1024 ** 3
