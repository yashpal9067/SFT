import os
from typing import List

from ..conversation import Conversation, Message


def load_txt_conversations(file_path: str) -> List[Conversation]:
    conversations: List[Conversation] = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TXT dataset not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            user_text, assistant_text = parts[0].strip(), parts[1].strip()
            if not user_text or not assistant_text:
                continue
            conv = Conversation(messages=[Message(role="user", content=user_text), Message(role="assistant", content=assistant_text)])
            conversations.append(conv)

    return conversations
