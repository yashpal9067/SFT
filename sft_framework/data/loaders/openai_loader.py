import json
import os
from typing import Any, Dict, List

from ..conversation import Conversation, Message


def _load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _normalize_openai(entry: Dict[str, Any]) -> Conversation:
    messages = []
    for turn in entry.get("messages", []):
        role = turn.get("role", "user").lower()
        content = turn.get("content", "")
        if isinstance(content, dict) and "content" in content:
            content = content["content"]
        messages.append(Message(role=role, content=content))
    return Conversation(messages=messages)


def load_openai_conversations(file_path: str) -> List[Conversation]:
    data = _load_json(file_path)
    conversations: List[Conversation] = []

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "messages" in data:
        entries = [data]
    else:
        entries = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        conversations.append(_normalize_openai(entry))

    return conversations
