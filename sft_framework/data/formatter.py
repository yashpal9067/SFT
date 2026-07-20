from typing import List, Optional, Tuple

from .conversation import Conversation, Message


class ChatTemplate:
    def __init__(self, name: str = "plain", custom_template: Optional[str] = None):
        self.name = name.lower()
        self.custom_template = custom_template

    def format_conversation(self, conversation: Conversation) -> Tuple[str, List[str]]:
        chunks: List[str] = []
        assistant_chunks: List[str] = []

        for message in conversation.messages:
            formatted = self._format_message(message)
            chunks.append(formatted)
            if message.role == "assistant":
                assistant_chunks.append(formatted)

        text = "".join(chunks)
        return text, assistant_chunks

    def _format_message(self, message: Message) -> str:
        role = message.role.lower()
        content = message.content.strip()

        if self.name == "chatml":
            if role == "assistant":
                return f"<|im_start|>assistant\n{content}<|im_end|>\n"
            if role == "user":
                return f"<|im_start|>user\n{content}<|im_end|>\n"
            return f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if self.name == "llama2":
            if role == "assistant":
                return f"[/INST] {content} </s>\n"
            if role == "user":
                return f"<s>[INST] {content} [/INST] "
            if role == "system":
                return f"<s>[INST] <<SYS>>\n{content}\n<</SYS>> "
            return f"<s>[INST] {content} [/INST] "

        if self.name == "llama3":
            if role == "assistant":
                return f"[/INST]\n{content}</s>\n"
            if role == "user":
                return f"<s>[INST] {content} [/INST] "
            if role == "system":
                return f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n" 
            return f"<s>[INST] {content} [/INST] "

        if self.name == "vicuna":
            if role == "assistant":
                return f"ASSISTANT: {content}\n"
            if role == "user":
                return f"USER: {content}\n"
            return f"{role.upper()}: {content}\n"

        if self.name == "alpaca":
            if role == "assistant":
                return f"### Response:\n{content}\n"
            if role == "user":
                return f"### Instruction:\n{content}\n"
            return f"### {role.capitalize()}:\n{content}\n"

        if self.name == "mistral":
            if role == "assistant":
                return f"### Assistant:\n{content}\n"
            if role == "user":
                return f"### Human:\n{content}\n"
            return f"### {role.capitalize()}:\n{content}\n"

        if self.name == "plain":
            if role == "assistant":
                return f"Assistant: {content}\n"
            if role == "user":
                return f"User: {content}\n"
            return f"{role.capitalize()}: {content}\n"

        if self.name == "custom" and self.custom_template is not None:
            if role == "assistant":
                return self.custom_template.replace("{assistant}", content).replace("{role}", role)
            return self.custom_template.replace("{user}", content).replace("{role}", role)

        return f"User: {content}\n" if role == "user" else f"Assistant: {content}\n"
