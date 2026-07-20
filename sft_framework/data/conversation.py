from dataclasses import dataclass, field
from typing import List


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Conversation:
    messages: List[Message] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)
