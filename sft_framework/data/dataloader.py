from typing import Optional

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .collator import DataCollatorForSFT


def create_dataloader(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    pad_to_multiple_of: Optional[int] = 8,
) -> DataLoader:
    collator = DataCollatorForSFT(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
