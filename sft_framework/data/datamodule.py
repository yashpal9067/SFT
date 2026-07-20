from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .conversation import Conversation
from .formatter import ChatTemplate
from .loaders import load_dataset_by_type


class SFTDataset(Dataset):
    def __init__(
        self,
        conversations: List[Conversation],
        tokenizer: PreTrainedTokenizerBase,
        template: ChatTemplate,
        max_length: int = 512,
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversation = self.conversations[idx]
        text, assistant_segments = self.template.format_conversation(conversation)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors=None,
        )

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)

        labels = input_ids.clone()
        labels = self._mask_non_assistant_tokens(conversation, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_non_assistant_tokens(self, conversation: Conversation, labels: torch.Tensor) -> torch.Tensor:
        text = self.template.format_conversation(conversation)[0]
        tokenized = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=self.max_length)
        offsets = tokenized["offset_mapping"]
        assistant_spans = []
        cursor = 0

        for message in conversation.messages:
            start = text.find(self._format_role_prefix(message.role), cursor)
            if start == -1:
                continue
            end = start + len(self._format_role_prefix(message.role)) + len(message.content)
            if message.role == "assistant":
                assistant_spans.append((start, end))
            cursor = end

        if not assistant_spans:
            labels[:] = -100
            return labels

        for i, (begin, end) in enumerate(offsets):
            if begin == end:
                labels[i] = -100
                continue
            if not any(span_begin <= begin < span_end for span_begin, span_end in assistant_spans):
                labels[i] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def _format_role_prefix(self, role: str) -> str:
        role = role.lower()
        if self.template.name == "plain":
            return f"Assistant: " if role == "assistant" else f"User: " if role == "user" else f"{role.capitalize()}: "
        if self.template.name == "chatml":
            return f"<|im_start|>assistant\n" if role == "assistant" else f"<|im_start|>user\n"
        if self.template.name == "alpaca":
            return f"### Response:\n" if role == "assistant" else f"### Instruction:\n"
        if self.template.name == "vicuna":
            return f"ASSISTANT: " if role == "assistant" else f"USER: "
        if self.template.name == "llama2":
            return f"[/INST] " if role == "assistant" else f"<s>[INST] "
        if self.template.name == "mistral":
            return f"### Assistant:\n" if role == "assistant" else f"### Human:\n"
        if self.template.name == "custom" and self.template.custom_template:
            if role == "assistant":
                return self.template.custom_template.split("{assistant}")[0]
            return self.template.custom_template.split("{user}")[0]
        return f"{role.capitalize()}: "


class DataModule:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.template = ChatTemplate(name=config.template_name, custom_template=config.custom_template)

    def load_dataset(self, file_path: str, dataset_type: str) -> SFTDataset:
        conversations = load_dataset_by_type(dataset_type, file_path)
        return SFTDataset(
            conversations=conversations,
            tokenizer=self.tokenizer,
            template=self.template,
            max_length=self.config.max_length,
        )
