# coref_automl/tune.py
import os
import gc
import math
import random
import numpy as np
import optuna
import torch
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path
import inspect

from datasets import load_dataset, disable_caching
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)
from torch.optim import AdamW as TorchAdamW

from .coref_utils import is_noun
from .callbacks import LiveMetricsCallback
from .dashboard import BUS

disable_caching()

# ----------------------------
# 데이터 로딩/전처리 유틸
# ----------------------------
def load_ko_lambada_split(prefer=("validation", "dev", "test")):
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    last_err = None
    for sp in prefer:
        try:
            kwargs = dict(split=sp)
            if hf_token:
                kwargs["token"] = hf_token
            return load_dataset("thunder-research-group/SNU_Ko-LAMBADA", **kwargs)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Ko-LAMBADA split 로드 실패: {last_err}")

def build_eval_set_from_ko_lambada(split_prefer=("validation", "dev", "test"), limit=1000, seed=42):
    ds = load_ko_lambada_split(prefer=split_prefer)
    items = []
    N = len(ds)
    cut = int(N * 0.8)
    dev = ds.select(range(0, cut)) if cut > 0 else ds
    tried = 0
    for ex in dev:
        text = ex.get("text") or ex.get("context") or ""
        target = ex.get("target") or ex.get("answer") or ""
        if not text or not target:
            continue
        if text.strip().endswith(target):
            masked = text[: len(text) - len(target)] + "[MASK]"
        else:
            masked = text.replace(target, "[MASK]") if target in text else None
        if masked:
            items.append({"masked": masked, "target": target})
            if len(items) >= limit:
                break
        tried += 1
        if tried > N:
            break
    return items

@dataclass
class DynCollator(DataCollatorForLanguageModeling):
    min_prob: float = 0.10
    max_prob: float = 0.25
    max_length: int = 512
    def __call__(self, examples):
        import numpy as _np
        lengths = [len(ex["input_ids"]) for ex in examples]
        avg = float(_np.mean(lengths)) if lengths else 0.0
        self.mlm_probability = float(self.min_prob + (self.max_prob - self.min_prob) * min(1.0, avg / self.max_length))
        return super().__call__(examples)

def build_mlm_dataset(
    tokenizer,
    max_length=512,
    mlm_corpus="wikimedia/wikipedia",
    subset="20231101.ko",
    split="train",
    limit=None,
):
    ds = load_dataset(mlm_corpus, subset, split=split)
    if limit is not None and 0 < limit < len(ds):
        ds = ds.select(range(limit))
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    keep = {"text"}
    rem = [c for c in ds.column_names if c not in keep]
    tokenized = ds.map(tok, batched=True, remove_columns=rem)
    return tokenized

# ----------------------------
# 평가 지표
# ----------------------------
def topk_noun_acc(masked_texts: List[Dict[str, Any]], fill_pipe, k=5) -> float:
    ok = 0
    tot = 0
    for item in masked_texts:
        mt = item["masked"]
        gold = item["target"]
        preds = fill_pipe(mt, top_k=max(50, k))
        cand = []
        for p in preds:
            token_str = p["token_str"].strip().replace("##", "")
            if token_str and is_noun(token_str):
                cand.append(token_str)
            if len(cand) >= k:
                break
        tot += 1
        if gold in cand or any(gold in c or c in gold for c in cand):
            ok += 1
    return ok / max(1, tot)

# ----------------------------
# TrainingArguments 호환 작성기
# ----------------------------
import inspect as _inspect
def make_training_args(out_dir: str, base_kwargs: dict) -> TrainingArguments:
    sig = _inspect.signature(TrainingArguments.__init__)
    allowed = set(p.name for p in sig.parameters.values())
    kwargs = {k: v for k, v in base_kwargs.items() if k in allowed}
    for k in ["evaluation_strategy", "eval_strategy", "eval_steps", "load_best_model_at_end", "metric_for_best_model"]:
        kwargs.pop(k, None)
    kwargs["output_dir"] = out_dir
    return TrainingArguments(**kwargs)

def bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major = torch.cuda.get_device_capability(0)[0]
        return major >= 8
    except Exception:
        return False

# ----------------------------
# LLRD 옵티마이저
# ----------------------------
from torch.optim import AdamW as TorchAdamW
def create_llrd_optimizer(model, lr: float, wd: float, llrd: float) -> TorchAdamW:
    no_decay = ["bias", "LayerNorm.weight"]
    named = list(model.named_parameters())
    total_layers = 0
    for n, _ in named:
        if "encoder.layer." in n:
            try:
                idx = int(n.split("encoder.layer.")[1].split(".")[0])
                total_layers = max(total_layers, idx + 1)
            except Exception:
                pass
    if total_layers == 0:
        total_layers = 12
    groups = []
    for n, p in named:
        lr_here = lr
        if "encoder.layer." in n and llrd < 0.999:
            try:
                k = int(n.split("encoder.layer.")[1].split(".")[0])
            except Exception:
                k = total_layers - 1
            lr_here = lr * (llrd ** (total_layers - 1 - k))
        wd_here = 0.0 if any(nd in n for nd in no_decay) else wd
        groups.append({"params": [p], "weight_decay": wd_here, "lr": lr_here})
    return TorchAdamW(groups)

# ----------------------------
# 안전 학습 (오류 자동 폴백)
# ----------------------------
def safe_train_once(
    model_name: str,
    trial_id: int,
    lr: float, wd: float, warmup: float,
    min_prob: float, max_prob: float, seq_len: int,
    grad_ckpt: bool, llrd: float,
    per_device_bs: int, grad_acc: int,
    train_limit: int | None, seed: int,
    out_dir: str,
) -> Tuple[AutoModelForMaskedLM, AutoTokenizer, Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    else:
        model.config.use_cache = False

    tokenized = build_mlm_dataset(tokenizer, max_length=seq_len, split="train", limit=train_limit)
    collator = DynCollator(tokenizer=tokenizer, mlm=True, min_prob=min_prob, max_prob=max_prob, max_length=seq_len)

    base_args = dict(
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=max(2, per_device_bs),
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        weight_decay=wd,
        warmup_ratio=warmup,
        num_train_epochs=1,
        logging_steps=50,
        save_strategy="no",
        bf16=bf16_supported(),
        report_to=[],
        seed=seed,
        dataloader_drop_last=False,
    )
    args = make_training_args(out_dir, base_args)

    optimizer = create_llrd_optimizer(model, lr, wd, llrd)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
    )

    # === 총 스텝 계산 & HP 메타데이터 구성 ===
    train_len = len(tokenized)
    bs = int(getattr(args, "per_device_train_batch_size", 1))
    ga = int(getattr(args, "gradient_accumulation_steps", 1))
    epochs = int(getattr(args, "num_train_epochs", 1))
    train_batches = math.ceil(train_len / max(1, bs))
    steps_per_epoch = math.ceil(train_batches / max(1, ga))
    total_steps = int(steps_per_epoch * epochs)

    hp = {
        "lr": float(lr),
        "weight_decay": float(wd),
        "warmup_ratio": float(warmup),
        "min_prob": float(min_prob),
        "max_prob": float(max_prob),
        "seq_len": int(seq_len),
        "grad_ckpt": bool(grad_ckpt),
        "llrd": float(llrd),
        "per_device_bs": int(bs),
        "grad_acc": int(ga),
        "bf16": bool(bf16_supported()),
        "train_limit": int(train_limit) if train_limit else None,
        "dataset_len": int(train_len),
        "epochs": int(epochs),
    }

    # 실시간 콜백: 총 스텝/HP/월드사이즈 포함
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    trainer.add_callback(
        LiveMetricsCallback(
            model_name=model_name,
            trial_id=trial_id,
            total_trials=None,         # run_study에서 설정
            total_steps=total_steps,
            hp=hp,
            world_size=world_size,
        )
    )

    trainer.train()
    return model, tokenizer, {"args": args, "bs": bs, "grad_acc": ga, "grad_ckpt": grad_ckpt, "total_steps": total_steps, "dataset_len": train_len}

# ----------------------------
# Optuna Objective
# ----------------------------
def objective(trial: optuna.Trial, model_name: str, planned_trials: int, train_limit: int | None = None, seed: int = 42):
    lr = trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True)
    wd = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    min_prob = trial.suggest_float("min_prob", 0.05, 0.15)
    max_prob = trial.suggest_float("max_prob", 0.20, 0.35)
    seq_len = trial.suggest_categorical("max_length", [256, 384, 512])
    grad_ckpt = trial.suggest_categorical("gradient_checkpo_

