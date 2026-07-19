import json
import os
from dataclasses import asdict

from sft_framework.configs import TrainConfig, default_train_config
from sft_framework.data.datamodule import DataModule
from sft_framework.data.dataloader import create_dataloader
from sft_framework.models import load_model_and_tokenizer
from sft_framework.trainer import SFTTrainer, set_seed, get_optimizer, setup_logger


def run_training(config):
    logger = setup_logger(config.log_file)
    set_seed(config.seed)

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=config.model_name_or_path,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
    )

    data_module = DataModule(tokenizer=tokenizer, config=config)
    train_dataset = data_module.load_dataset(config.train_file, config.dataset_type)
    eval_dataset = None
    if config.validation_file:
        eval_dataset = data_module.load_dataset(config.validation_file, config.dataset_type)

    train_dataloader = create_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = create_dataloader(
            eval_dataset,
            tokenizer=tokenizer,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        )

    optimizer = get_optimizer(
        optimizer_name=config.optimizer,
        parameters=model.parameters(),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        config=config,
        logger=logger,
    )

    if config.resume_from_checkpoint:
        trainer.load_checkpoint(config.resume_from_checkpoint)

    trainer.train()


if __name__ == "__main__":
    default_config = default_train_config()
    config_path = os.path.join(os.path.dirname(__file__), "train_config.json")

    if os.path.exists(config_path):
        config = TrainConfig.load_json(config_path)
    else:
        config = default_config
        config.train_file = os.path.join(os.path.dirname(__file__), "data", "sample_alpaca.json")
        config.validation_file = None
        config.dataset_type = "alpaca"
        config.template_name = "alpaca"
        config.train_batch_size = 2
        config.eval_batch_size = 2
        config.num_epochs = 1
        config.save_every_steps = 10
        config.eval_every_steps = 5
        config.logging_steps = 1
        config.checkpoint_dir = os.path.join(os.path.dirname(__file__), "outputs")
        config.log_file = os.path.join(config.checkpoint_dir, "train.log")

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        config.save_json(config_path)

    run_training(config)
