import json
import os
from typing import Any, Dict, List

from ..conversation import Conversation, Message


def _load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _normalize_instruction(entry: Dict[str, Any]) -> Conversation:
    instruction = entry.get("instruction", "")
    extra_input = entry.get("input", "")
    output = entry.get("output", "")

    user_content = instruction
    if extra_input:
        user_content = f"{instruction}\n\n{extra_input}".strip()

    return Conversation(messages=[Message(role="user", content=user_content), Message(role="assistant", content=output)])


def load_alpaca_conversations(file_path: str) -> List[Conversation]:
    data = _load_json(file_path)
    conversations: List[Conversation] = []

    if isinstance(data, dict) and "data" in data:
        entries = data["data"]
    else:
        entries = data

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        conversations.append(_normalize_instruction(entry))

    return conversations
