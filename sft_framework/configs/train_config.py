import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    model_name_or_path: str = "gpt2"
    tokenizer_name_or_path: Optional[str] = None
    train_file: str = "./data/sample_alpaca.json"
    validation_file: Optional[str] = None
    dataset_type: str = "alpaca"
    template_name: str = "alpaca"
    custom_template: Optional[str] = None
    max_length: int = 512
    train_batch_size: int = 4
    eval_batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    scheduler: str = "linear"
    warmup_steps: int = 100
    total_training_steps: Optional[int] = None
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_every_steps: int = 500
    eval_every_steps: int = 250
    logging_steps: int = 50
    checkpoint_dir: str = "./outputs"
    log_file: Optional[str] = None
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    resume_from_checkpoint: Optional[str] = None
    fp16: bool = False
    use_cuda: bool = True

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TrainConfig":
        parsed = {}
        field_names = {f.name for f in fields(cls)}
        for key, value in values.items():
            if key in field_names:
                parsed[key] = value
        return cls(**parsed)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def load_json(cls, path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return cls.from_dict(data)
        raise ValueError("Config file must contain a JSON object.")

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def default_train_config() -> TrainConfig:
    return TrainConfig()
