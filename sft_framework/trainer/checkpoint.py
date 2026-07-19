import json
import os
from dataclasses import asdict
from typing import Optional

import torch
from transformers import PreTrainedModel


def save_checkpoint(
    output_dir: str,
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
    config,
) -> str:
    checkpoint_name = f"checkpoint-step-{global_step}" if global_step else f"checkpoint-epoch-{epoch}"
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.save_pretrained(checkpoint_dir)

    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    torch.save(optimizer.state_dict(), optimizer_path)

    if scheduler is not None:
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(scheduler.state_dict(), scheduler_path)

    trainer_state = {
        "epoch": epoch,
        "global_step": global_step,
    }
    with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)

    config_path = os.path.join(checkpoint_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: str,
    model: PreTrainedModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location="cpu"))

    if optimizer is not None:
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))

    if scheduler is not None:
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))

    trainer_state = {"epoch": 0, "global_step": 0}
    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            trainer_state = json.load(f)

    return trainer_state
