import math
import os
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer
from tqdm.auto import tqdm

from .checkpoint import save_checkpoint, load_checkpoint
from .scheduler import create_scheduler
from .utils import get_gpu_memory


class SFTTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader=None,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
        config=None,
        logger=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
        self.model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 and self.device.type == "cuda" else None

        self.global_step = 0
        self.start_epoch = 0

        if self.optimizer is None:
            raise ValueError("Optimizer must be provided to SFTTrainer.")
        if self.scheduler is None:
            total_steps = self.config.total_training_steps
            if total_steps is None:
                total_steps = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps) * self.config.num_epochs
            self.scheduler = create_scheduler(
                scheduler_type=self.config.scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps,
            )

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        loss = loss / self.config.gradient_accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach() * self.config.gradient_accumulation_steps

    def optimizer_step(self) -> float:
        grad_norm = self._compute_grad_norm()
        
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return grad_norm

    def evaluate(self) -> Dict[str, Any]:
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Validation", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
        return {"eval_loss": avg_loss, "perplexity": perplexity}

    def save_checkpoint(self, step: Optional[int] = None) -> str:
        return save_checkpoint(
            output_dir=self.config.checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.start_epoch,
            global_step=self.global_step if step is None else step,
            config=self.config,
        )

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        trainer_state = load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.start_epoch = trainer_state.get("epoch", 0)
        self.global_step = trainer_state.get("global_step", 0)
        if self.logger:
            self.logger.info(f"Resumed checkpoint at epoch {self.start_epoch}, step {self.global_step}")

    def train(self) -> None:
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Batch size: {self.config.train_batch_size}, epochs: {self.config.num_epochs}")
        self.logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")

        self.model.train()
        global_step = self.global_step

        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            tokens_processed = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for step, batch in enumerate(progress_bar):
                batch_loss = self.training_step(batch)
                epoch_loss += batch_loss.item()
                epoch_steps += 1
                tokens_processed += batch["attention_mask"].sum().item()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    grad_norm = self.optimizer_step()
                    global_step += 1
                    self.global_step = global_step

                    lr = self.scheduler.get_last_lr()[0]
                    tokens_per_sec = tokens_processed / max(progress_bar.format_dict.get("elapsed", 1), 1)
                    gpu_mem = get_gpu_memory(self.device) if self.device.type == "cuda" else 0.0

                    progress_bar.set_postfix({
                        "loss": f"{batch_loss.item():.4f}",
                        "lr": f"{lr:.2e}",
                        "grad_norm": f"{grad_norm:.4f}",
                        "tokens/s": f"{tokens_per_sec:.1f}",
                        "gpu_gb": f"{gpu_mem:.2f}",
                    })

                    if global_step % self.config.eval_every_steps == 0 and self.eval_dataloader is not None:
                        eval_metrics = self.evaluate()
                        self.logger.info(f"Step {global_step} Eval loss={eval_metrics['eval_loss']:.4f} perplexity={eval_metrics['perplexity']:.2f}")
                        self.model.train()

                    if global_step % self.config.save_every_steps == 0:
                        checkpoint_dir = self.save_checkpoint(global_step)
                        self.logger.info(f"Saved checkpoint to {checkpoint_dir}")

                    if self.config.log_file and global_step % self.config.logging_steps == 0:
                        self.logger.info(
                            f"Epoch {epoch + 1} step {global_step} loss={batch_loss.item():.4f} lr={lr:.2e} grad_norm={grad_norm:.4f}"
                        )
                else:
                    progress_bar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

            if (len(self.train_dataloader) % self.config.gradient_accumulation_steps) != 0:
                self.optimizer_step()
                global_step += 1
                self.global_step = global_step

            average_epoch_loss = epoch_loss / max(epoch_steps, 1)
            self.logger.info(f"Epoch {epoch + 1} completed. average_loss={average_epoch_loss:.4f}")
            self.start_epoch = epoch + 1
            self.save_checkpoint()

        if self.eval_dataloader is not None:
            eval_metrics = self.evaluate()
            self.logger.info(f"Final evaluation: {eval_metrics}")

        self.logger.info("Training completed.")

    def _compute_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
