from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForSFT:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]

        batch_max_length = max(tensor.size(0) for tensor in input_ids)
        if self.pad_to_multiple_of is not None:
            remainder = batch_max_length % self.pad_to_multiple_of
            if remainder != 0:
                batch_max_length += self.pad_to_multiple_of - remainder

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, label in zip(input_ids, attention_mask, labels):
            pad_length = batch_max_length - ids.size(0)
            if pad_length > 0:
                pad_id = self.tokenizer.pad_token_id
                ids = torch.cat([ids, torch.full((pad_length,), pad_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
                label = torch.cat([label, torch.full((pad_length,), self.label_pad_token_id, dtype=label.dtype)])
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
            padded_labels.append(label)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }
