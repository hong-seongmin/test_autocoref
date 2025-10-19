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
    build_real_coref_eval_set,
    build_eval_from_lambada,
    eval_real_coref_top1,
    eval_real_coref_top5,
    eval_lambada_topk,
)


def recommended_hparams(seq_len: int):
    if seq_len <= 1536:
        return {
            "lr": 8.0e-5,
            "warmup_ratio": 0.16,
            "min_prob": 0.118,
            "max_prob": 0.36,
            "per_device_bs": 32,  # H100 80GB용 초기 배치 크기 증가 (8->32)
            "grad_acc": 4,  # gradient accumulation 감소 (16->4)
            "weight_decay": 0.045,
        }
    return {
        "lr": 2.3e-5,
        "warmup_ratio": 0.195,
        "min_prob": 0.136,
        "max_prob": 0.475,
        "per_device_bs": 16,  # 더 긴 시퀀스도 배치 크기 증가 (4->16)
        "grad_acc": 8,  # gradient accumulation 감소 (32->8)
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
        resume_from_step: int = 0,
    ):
        # 이미 완료된 step은 제외 (이어하기 지원)
        self._pending_steps: Set[int] = {s for s in eval_steps if s > resume_from_step}
        self._tokenizer = tokenizer
        self._seq_len = seq_len
        self._device = 0 if torch.cuda.is_available() else -1
        self._lbd = build_eval_from_lambada(limit=600)
        self._coref = build_real_coref_eval_set(limit=1600, max_seq_len=seq_len, seed=999)
        self._trainer: Optional[Trainer] = None
        self._steps_per_epoch = steps_per_epoch
        self._total_epochs = total_epochs
        self._log_file = output_dir / "half_epoch_eval.jsonl"
        # 이어하기: 로그 파일 보존 (삭제하지 않음)

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
        real1 = eval_real_coref_top1(fill, self._coref, mask_token=mask_token, batch_size=32, seq_len=self._seq_len)
        real5 = eval_real_coref_top5(fill, self._coref, mask_token=mask_token, batch_size=32, seq_len=self._seq_len)
        score = 0.4 * real1 + 0.3 * real5 + 0.3 * lbd_top1

        if step_value is not None and self._steps_per_epoch > 0:
            epoch = step_value / self._steps_per_epoch
            label = f"{epoch:.1f} epoch"
        else:
            epoch = self._total_epochs
            label = "final"

        print(
            f"[Half-epoch Eval] {label} | "
            f"LAMBADA@1={lbd_top1:.4f} | Real@1={real1:.4f} | Real@5={real5:.4f} | score={score:.4f}"
        )

        self._log_record(
            {
                "epoch": epoch,
                "step": step_value,
                "lambada_top1": lbd_top1,
                "real1": real1,
                "real5": real5,
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


def find_last_checkpoint(output_dir: Path) -> Optional[str]:
    """output_dir에서 가장 최근 checkpoint-* 디렉토리 찾기"""
    if not output_dir.exists():
        return None
    checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    # checkpoint-143 → 143 추출하여 정렬
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return str(checkpoints[-1])


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

    # Gradient checkpointing으로 메모리 절약 (더 큰 배치 사용 가능)
    model.gradient_checkpointing_enable()

    if args.seq_len > model.config.max_position_embeddings:
        print(f"→ Resizing position embeddings from {model.config.max_position_embeddings} to {args.seq_len}")

        # Detect model type and resize accordingly
        if hasattr(model, 'deberta'):
            # DeBERTa: uses relative position embeddings
            print("   Model type: DeBERTa")
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

        elif hasattr(model, 'roberta'):
            # RoBERTa: uses absolute position embeddings
            print("   Model type: RoBERTa")
            position_embed = model.roberta.embeddings.position_embeddings
            old_num, dim = position_embed.weight.shape
            # RoBERTa uses position_ids starting from padding_idx+1 (2)
            # So we need seq_len + padding positions
            new_num_positions = args.seq_len + 2  # +2 for padding (0, 1)
            new_embed = torch.nn.Embedding(new_num_positions, dim)
            new_embed.weight.data[:old_num] = position_embed.weight.data.clone()
            if new_num_positions > old_num:
                # Repeat the last position embedding for new positions
                new_embed.weight.data[old_num:] = position_embed.weight.data[-1:].repeat(new_num_positions - old_num, 1)
            model.roberta.embeddings.position_embeddings = new_embed.to(model.device)
            model.config.max_position_embeddings = args.seq_len
            # Also update the position_ids buffer size
            model.roberta.embeddings.register_buffer(
                "position_ids",
                torch.arange(new_num_positions).expand((1, -1)),
                persistent=False
            )
            print(f"   RoBERTa position embeddings: {old_num} → {new_num_positions}")

        else:
            raise NotImplementedError(
                f"Sequence length resizing not supported for model type: {model.config.model_type}"
            )

        print(f"✓ Position embeddings resized successfully")

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

    # 자동 체크포인트 탐색 (이어하기)
    checkpoint_path = find_last_checkpoint(output_dir)
    resume_from_step = 0
    if checkpoint_path:
        # checkpoint-143 → 143 추출
        resume_from_step = int(Path(checkpoint_path).name.split("-")[1])
        print(f"✅ 체크포인트 발견: {checkpoint_path}")
        print(f"   Step {resume_from_step}부터 이어하기...")
    else:
        print("ℹ️  체크포인트 없음, 처음부터 시작")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_acc,
        auto_find_batch_size=True,  # GPU 메모리에 맞게 배치 크기 자동 조정
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
        resume_from_step=resume_from_step,
    )
    callback.set_trainer(trainer)
    trainer.add_callback(callback)

    # 체크포인트에서 이어하기
    trainer.train(resume_from_checkpoint=checkpoint_path)


if __name__ == "__main__":
    main()
