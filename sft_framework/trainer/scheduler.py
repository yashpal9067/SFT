from transformers import get_scheduler
from torch.optim import Optimizer


def create_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "constant":
        return get_scheduler(
            name="constant",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    if scheduler_type == "cosine":
        return get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    if scheduler_type == "linear":
        return get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
