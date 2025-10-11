import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Set

import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    TrainerCallback,
)

from coref_automl.tune import (
    DynCollator,
    build_coref_eval_set,
    build_eval_from_lambada,
    eval_coref_f1,
    eval_coref_recall_topk,
    eval_lambada_topk,
)


def recommended_hparams(seq_len: int):
    if seq_len <= 1536:
        return {
            "lr": 8.0e-5,
            "warmup_ratio": 0.16,
            "min_prob": 0.118,
            "max_prob": 0.36,
            "per_device_bs": 8,
            "grad_acc": 16,
            "weight_decay": 0.045,
        }
    return {
        "lr": 2.3e-5,
        "warmup_ratio": 0.195,
        "min_prob": 0.136,
        "max_prob": 0.475,
        "per_device_bs": 4,
        "grad_acc": 32,
        "weight_decay": 0.046,
    }


class HalfEpochEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_steps: Sequence[int],
        tokenizer,
        seq_len: int,
        output_dir: Path,
        steps_per_epoch: int,
        total_epochs: float,
    ):
        self._pending_steps: Set[int] = set(eval_steps)
        self._tokenizer = tokenizer
        self._seq_len = seq_len
        self._device = 0 if torch.cuda.is_available() else -1
        self._lbd = build_eval_from_lambada(limit=600)
        self._coref = build_coref_eval_set(limit=800, max_seq_len=seq_len)
        self._trainer: Optional[Trainer] = None
        self._steps_per_epoch = steps_per_epoch
        self._total_epochs = total_epochs
        self._log_file = output_dir / "half_epoch_eval.jsonl"
        if self._log_file.exists():
            self._log_file.unlink()

    def on_train_begin(self, args, state, control, **kwargs):
        self._trainer = kwargs.get("trainer")
        return control

    def set_trainer(self, trainer: Trainer):
        self._trainer = trainer

    def _log_record(self, record: dict):
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _run_eval(self, trainer: Trainer, step_value: Optional[int]):
        model = trainer.model
        was_training = model.training
        model.eval()
        fill = pipeline(
            "fill-mask",
            model=model,
            tokenizer=self._tokenizer,
            device=self._device,
        )
        mask_token = self._tokenizer.mask_token or "[MASK]"
        lbd_top1 = eval_lambada_topk(fill, self._lbd, mask_token=mask_token, k=1, batch_size=32, seq_len=self._seq_len)
        coref_f1 = eval_coref_f1(fill, self._coref, mask_token=mask_token, k=5, batch_size=32, seq_len=self._seq_len)
        coref_top5 = eval_coref_recall_topk(fill, self._coref, mask_token=mask_token, k=5, batch_size=32, seq_len=self._seq_len)
        score = 0.4 * coref_f1 + 0.3 * coref_top5 + 0.3 * lbd_top1

        if step_value is not None and self._steps_per_epoch > 0:
            epoch = step_value / self._steps_per_epoch
            label = f"{epoch:.1f} epoch"
        else:
            epoch = self._total_epochs
            label = "final"

        print(
            f"[Half-epoch Eval] {label} | "
            f"LAMBADA@1={lbd_top1:.4f} | Coref F1={coref_f1:.4f} | Coref@5={coref_top5:.4f} | score={score:.4f}"
        )

        self._log_record(
            {
                "epoch": epoch,
                "step": step_value,
                "lambada_top1": lbd_top1,
                "coref_f1": coref_f1,
                "coref_top5": coref_top5,
                "score": score,
            }
        )

        if was_training:
            model.train()

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step in self._pending_steps:
            self._pending_steps.remove(step)
            trainer = self._trainer or kwargs.get("trainer")
            if trainer is not None:
                self._run_eval(trainer, step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._pending_steps:
            trainer = self._trainer or kwargs.get("trainer")
            if trainer is not None:
                self._run_eval(trainer, None)
        return control


def compute_eval_steps(
    total_samples: int,
    per_device_bs: int,
    grad_acc: int,
    epochs: float,
    eval_interval: float,
) -> (List[int], int):
    steps_per_epoch = math.ceil(total_samples / max(1, per_device_bs * grad_acc))
    total_steps = math.ceil(steps_per_epoch * epochs)
    eval_steps = []
    current = eval_interval
    while current <= epochs + 1e-9:
        step = int(round(current * steps_per_epoch))
        if 0 < step <= total_steps:
            eval_steps.append(step)
        current += eval_interval
    return sorted(set(eval_steps)), steps_per_epoch


def load_dataset_from_disk(path: str) -> Dataset:
    dataset = Dataset.load_from_disk(path)
    required = {"input_ids", "attention_mask"}
    missing = required - set(dataset.features.keys())
    if missing:
        raise ValueError(f"Dataset at {path} is missing required columns: {missing}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Run long-sequence MLM training on combined datasets with half-epoch evals.")
    parser.add_argument("--model", default="kakaobank/kf-deberta-base")
    parser.add_argument("--dataset", required=True, help="Path to Dataset.load_from_disk directory")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--eval-interval", type=float, default=0.5)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--min-prob", type=float)
    parser.add_argument("--max-prob", type=float)
    parser.add_argument("--per-device-bs", type=int)
    parser.add_argument("--grad-acc", type=int)
    parser.add_argument("--output-dir", default="./runs/combined_experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    defaults = recommended_hparams(args.seq_len)
    lr = args.lr or defaults["lr"]
    warmup_ratio = args.warmup_ratio or defaults["warmup_ratio"]
    min_prob = args.min_prob or defaults["min_prob"]
    max_prob = args.max_prob or defaults["max_prob"]
    per_device_bs = args.per_device_bs or defaults["per_device_bs"]
    grad_acc = args.grad_acc or defaults["grad_acc"]
    weight_decay = defaults["weight_decay"]

    dataset = load_dataset_from_disk(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    if args.seq_len > model.config.max_position_embeddings:
        # DebertaV2 uses relative position embeddings stored in encoder.rel_embeddings.
        position_embed = getattr(model.deberta.embeddings, "position_embeddings", None)
        if position_embed is not None:
            old_num, dim = position_embed.weight.shape
            new_embed = torch.nn.Embedding(args.seq_len, dim)
            new_embed.weight.data[:old_num] = position_embed.weight.data.clone()
            if args.seq_len > old_num:
                new_embed.weight.data[old_num:] = position_embed.weight.data[-1:].repeat(args.seq_len - old_num, 1)
            model.deberta.embeddings.position_embeddings = new_embed.to(model.device)

        rel_embeddings = getattr(model.deberta.encoder, "rel_embeddings", None)
        if rel_embeddings is None:
            raise NotImplementedError(
                "Deberta model does not expose rel_embeddings; manual resize not supported."
            )
        old_rel_num, rel_dim = rel_embeddings.weight.shape
        new_rel = torch.nn.Embedding(args.seq_len, rel_dim)
        new_rel.weight.data[:old_rel_num] = rel_embeddings.weight.data.clone()
        if args.seq_len > old_rel_num:
            new_rel.weight.data[old_rel_num:] = rel_embeddings.weight.data[-1:].repeat(args.seq_len - old_rel_num, 1)
        model.deberta.encoder.rel_embeddings = new_rel.to(model.device)
        model.config.max_position_embeddings = args.seq_len

    collator = DynCollator(
        tokenizer=tokenizer,
        mlm=True,
        min_prob=min_prob,
        max_prob=max_prob,
        max_length=args.seq_len,
    )

    eval_steps, steps_per_epoch = compute_eval_steps(
        total_samples=len(dataset),
        per_device_bs=per_device_bs,
        grad_acc=grad_acc,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        num_train_epochs=args.epochs,
        max_grad_norm=0.5 if args.seq_len > 1024 else 1.0,
        logging_steps=max(1, eval_steps[0] if eval_steps else 50),
        eval_strategy="no",
        save_strategy="steps",
        save_steps=max(1, steps_per_epoch // 2),
        save_total_limit=None,
        report_to=[],
        seed=args.seed,
        dataloader_drop_last=False,
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    callback = HalfEpochEvalCallback(
        eval_steps=eval_steps,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        output_dir=output_dir,
        steps_per_epoch=steps_per_epoch,
        total_epochs=args.epochs,
    )
    callback.set_trainer(trainer)
    trainer.add_callback(callback)

    trainer.train()


if __name__ == "__main__":
    main()
