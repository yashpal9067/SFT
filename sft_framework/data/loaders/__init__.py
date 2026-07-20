from .alpaca_loader import load_alpaca_conversations
from .openai_loader import load_openai_conversations
from .sharegpt_loader import load_sharegpt_conversations
from .txt_loader import load_txt_conversations


def load_dataset_by_type(dataset_type: str, file_path: str):
    dataset_type = dataset_type.lower()
    if dataset_type == "txt":
        return load_txt_conversations(file_path)
    if dataset_type == "alpaca":
        return load_alpaca_conversations(file_path)
    if dataset_type == "sharegpt":
        return load_sharegpt_conversations(file_path)
    if dataset_type == "openai":
        return load_openai_conversations(file_path)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
