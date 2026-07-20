import json
import os
from typing import Any, Dict, List

from ..conversation import Conversation, Message


def _load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _normalize_sharegpt(entry: Dict[str, Any]) -> Conversation:
    messages = []
    for turn in entry.get("conversations", []):
        role = turn.get("from", "user").lower()
        if role == "gpt":
            role = "assistant"
        messages.append(Message(role=role, content=turn.get("value", "")))
    return Conversation(messages=messages)


def load_sharegpt_conversations(file_path: str) -> List[Conversation]:
    data = _load_json(file_path)
    conversations: List[Conversation] = []

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "conversations" in data:
        entries = [data]
    else:
        entries = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        conversations.append(_normalize_sharegpt(entry))

    return conversations
