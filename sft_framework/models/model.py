from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    pad_token_id: Optional[int] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    if pad_token_id is not None:
        tokenizer.pad_token_id = pad_token_id
        model.config.pad_token_id = pad_token_id

    return model, tokenizer
