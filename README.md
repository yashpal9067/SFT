# SFT Framework

This repository contains a PyTorch-based supervised fine-tuning (SFT) framework for conversational models. It is intentionally built without Hugging Face `Trainer`, `TRL`, or other high-level training abstractions.

## Project Structure

- `sft_framework/configs/train_config.py` - training configuration dataclass.
- `sft_framework/data/conversation.py` - unified conversation/message representation.
- `sft_framework/data/formatter.py` - chat template formatting for multiple styles.
- `sft_framework/data/loaders/` - loaders for TXT, Alpaca, ShareGPT, and OpenAI messages.
- `sft_framework/data/datamodule.py` - dataset wrapper and tokenizer integration.
- `sft_framework/data/collator.py` - dynamic padding and assistant-only label masking.
- `sft_framework/data/dataloader.py` - PyTorch `DataLoader` factory.
- `sft_framework/models/model.py` - model and tokenizer loading.
- `sft_framework/trainer/` - training loop, scheduler, checkpointing, logging, and utilities.
- `train.py` - example entrypoint for launching training.

## Configuration

The framework uses `train_config.json` for training configuration. Two example configs are provided:

- **`train_config.json`** (default) - Sample/debug config with 3 examples for quick testing
- **`train_config_full_alpaca.json`** - Full Alpaca dataset config (52K examples, optimized hyperparameters)

To switch between them:
```bash
# Use sample dataset (default)
python train.py

# Use full Alpaca dataset
cp train_config_full_alpaca.json train_config.json
python train.py
```

Or edit `train_config.json` directly to customize parameters like `train_file`, `batch_size`, `num_epochs`, `learning_rate`, etc.

## Supported Dataset Formats

- TXT: `User<TAB>Assistant`
- Alpaca JSON
- ShareGPT JSON
- OpenAI messages JSON

## Supported Templates

- `chatml`
- `llama2`
- `llama3`
- `vicuna`
- `alpaca`
- `mistral`
- `plain`
- `custom`
