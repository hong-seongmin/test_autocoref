# coref_automl/tune.py
from __future__ import annotations
import os
import gc
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import optuna
import torch
from datasets import load_dataset, disable_caching
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)

from .coref_utils import is_noun
from .callback import LiveMetricsCallback
from .bus import BUS

disable_caching()  # 캐싱 비활성화(요구사항)

CANDIDATE_MODELS = [
    "kakaobank/kf-deberta-base",
    "kykim/bert-kor-base",
    "google-bert/bert-base-multilingual-cased",
]

MASK_TOKEN_FALLBACK = "[MASK]"


# ────────────────────────────────────────────────────────────────────────────────
# Ko-LAMBADA 로더 (gated → token 자동 사용)
# ────────────────────────────────────────────────────────────────────────────────
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


def build_eval_from_lambada(limit=1000, seed=42) -> List[Dict[str, str]]:
    ds = load_ko_lambada_split(prefer=("validation", "dev", "test"))
    rnd = random.Random(seed)
    idxs = list(range(len(ds)))
    rnd.shuffle(idxs)
    items = []
    for i in idxs:
        ex = ds[i]
        text = ex.get("text") or ex.get("context") or ""
        target = ex.get("target") or ex.get("answer") or ""
        if not text or not target:
            continue
        if text.strip().endswith(target):
            masked = text[: len(text) - len(target)] + MASK_TOKEN_FALLBACK
        else:
            if target in text:
                masked = text.replace(target, MASK_TOKEN_FALLBACK, 1)
            else:
                masked = text + " " + MASK_TOKEN_FALLBACK
        items.append({"masked": masked, "target": target})
        if len(items) >= limit:
            break
    return items


# ────────────────────────────────────────────────────────────────────────────────
# Wikipedia 기반 Coref-Recall@k 평가 셋 (실데이터, 휴리스틱)
#  - 대명사(NP)를 [MASK]로, 이전 문맥(현재/직전 문장)의 '명사 집합'을 골드 후보로 정의
#  - 리콜: 예측 top-k 명사 중 하나라도 문맥 명사에 등장하면 hit
# ────────────────────────────────────────────────────────────────────────────────
def build_coref_eval_set(
    limit=2000,
    seed=123,
    max_seq_len: int = 512,
) -> List[Dict[str, Any]]:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    rnd = random.Random(seed)

    # 다중 데이터 소스
    sources = [
        ("wikimedia/wikipedia", "20231101.ko", "train"),
        ("klue", "ynat", "validation"),  # 뉴스 데이터
    ]

    items: List[Dict[str, Any]] = []
    scale = max(1, max_seq_len // 512)
    effective_limit = max(limit, int(limit * scale))
    target_per_source = max(1, math.ceil(effective_limit / len(sources)))
    limit_per_text = min(12, max(5, 5 + 2 * (scale - 1)))
    window_radius = min(3, max(1, scale - 1))
    per_source_counts = defaultdict(int)

    for source, subset, split in sources:
        try:
            ds = load_dataset(source, subset, split=split)
            idxs = list(range(len(ds)))
            rnd.shuffle(idxs)

            for i in idxs:
                if per_source_counts[source] >= target_per_source:
                    break
                if source == "wikimedia/wikipedia":
                    text = ds[i]["text"]
                elif source == "klue":
                    text = f"{ds[i]['title']} {ds[i].get('content', '')}".strip()
                else:
                    continue

                if not text or len(text) < 50:  # 최소 길이 증가
                    continue

                processed_items = process_coref_text(
                    text,
                    kiwi,
                    limit_per_text=limit_per_text,
                    window_radius=window_radius,
                )
                for item in processed_items:
                    if per_source_counts[source] >= target_per_source:
                        break
                    items.append(item)
                    per_source_counts[source] += 1
                if per_source_counts[source] >= target_per_source:
                    break
        except Exception as e:
            print(f"Warning: Failed to load {source}: {e}")
            continue

    # 품질 필터링
    filtered_items = []
    for item in items:
        if len(item["context_nouns"]) >= 2:  # 최소 2개 이상의 문맥 명사
            filtered_items.append(item)

    return filtered_items[:effective_limit]


def process_coref_text(
    text: str,
    kiwi,
    limit_per_text: int = 5,
    window_radius: int = 1,
) -> List[Dict[str, Any]]:
    """단일 텍스트에서 coref 샘플 생성"""
    import re

    def sent_split(text: str) -> List[str]:
        s = re.split(r"(?<=[.!?…])\s+", text)
        return [t.strip() for t in s if t.strip()]

    PRONOUN_POS = {"NP"}
    NOUN_POS = {"NNG", "NNP"}

    items = []
    sents = sent_split(text)

    for si, s in enumerate(sents):
        if len(items) >= limit_per_text:
            break

        toks = kiwi.tokenize(s)
        pron_candidates = []

        # 대명사 후보 찾기
        for ti, tk in enumerate(toks):
            if tk.tag in PRONOUN_POS:
                pron_candidates.append((ti, tk))

        for pron_idx, pron_tok in pron_candidates:
            # [MASK] 삽입
            start, end = pron_tok.start, pron_tok.end
            masked = s[:start] + MASK_TOKEN_FALLBACK + s[end:]

            # 확장된 문맥 (현재 문장 전체 + 주변 문장)
            context_parts = []
            for offset in range(-window_radius, window_radius + 1):
                idx = si + offset
                if 0 <= idx < len(sents):
                    context_parts.append(sents[idx])

            context_text = " ".join(context_parts)

            # 문맥 명사 추출
            nouns = set()
            for tkn in kiwi.tokenize(context_text):
                if tkn.tag in NOUN_POS:
                    nouns.add(tkn.form)

            if len(nouns) >= 2:  # 품질 기준 강화
                items.append({
                    "masked": masked,
                    "context_nouns": list(nouns),
                    "pronoun": pron_tok.form,
                    "sentence": s
                })

    return items


# ────────────────────────────────────────────────────────────────────────────────
# MLM 파인튜닝 데이터(실데이터: 위키)
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class DynCollator(DataCollatorForLanguageModeling):
    min_prob: float = 0.10
    max_prob: float = 0.25
    max_length: int = 512

    def __call__(self, examples):
        lengths = [len(ex["input_ids"]) for ex in examples]
        avg = float(np.mean(lengths)) if lengths else 0.0
        self.mlm_probability = float(self.min_prob + (self.max_prob - self.min_prob) * min(1.0, avg / self.max_length))
        return super().__call__(examples)


def build_mlm_dataset(tokenizer, max_length=512, split="train", limit=None):
    from datasets import concatenate_datasets

    datasets = []

    # 1. Wikipedia (기본)
    try:
        wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split=split)
        datasets.append(wiki_ds)
    except Exception as e:
        print(f"Warning: Failed to load Wikipedia: {e}")

    # 2. KLUE 뉴스 데이터
    try:
        klue_ds = load_dataset("klue", "ynat", split=split)
        # 뉴스 데이터를 텍스트 형태로 변환
        def convert_klue(example):
            return {"text": f"{example['title']} {example.get('content', '')}".strip()}
        klue_ds = klue_ds.map(convert_klue, remove_columns=klue_ds.column_names)
        datasets.append(klue_ds)
    except Exception as e:
        print(f"Warning: Failed to load KLUE: {e}")

    # 3. KorQuAD 데이터 (질문-답변)
    try:
        korquad_ds = load_dataset("squad_kor_v1", split=split)
        def convert_korquad(example):
            context = example.get("context", "")
            question = example.get("question", "")
            answer = example.get("answers", {}).get("text", [""])[0] if example.get("answers") else ""
            return {"text": f"{context} {question} {answer}".strip()}
        korquad_ds = korquad_ds.map(convert_korquad, remove_columns=korquad_ds.column_names)
        datasets.append(korquad_ds)
    except Exception as e:
        print(f"Warning: Failed to load KorQuAD: {e}")

    # 데이터 통합
    if not datasets:
        raise RuntimeError("No datasets could be loaded")

    combined_ds = concatenate_datasets(datasets)

    # 셔플 및 제한
    combined_ds = combined_ds.shuffle(seed=42)
    if limit is not None and len(combined_ds) > limit:
        combined_ds = combined_ds.select(range(limit))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = combined_ds.map(tok, batched=True, remove_columns=["text"])
    return tokenized


# ────────────────────────────────────────────────────────────────────────────────
# 길이 초과 방지(파이프라인에 truncation kwargs 금지 → 문자 윈도우로 잘라서 전달)
# ────────────────────────────────────────────────────────────────────────────────
def clip_around_mask(
    text: str,
    mask_token: str,
    left_chars: int = 200,
    right_chars: int = 200,
    seq_len: Optional[int] = None,
) -> str:
    try:
        i = text.index(mask_token)
        scale = 1.0
        if seq_len:
            scale = max(1.0, seq_len / 512)
        window_left = int(left_chars * scale)
        window_right = int(right_chars * scale)
        s = max(0, i - window_left)
        e = min(len(text), i + len(mask_token) + window_right)
        return text[s:e]
    except ValueError:
        # [MASK]가 없다면 그대로
        return text


# ────────────────────────────────────────────────────────────────────────────────
# 파이프라인 평가: LAMBADA 정확/ Coref 리콜 (배치/명사필터)
# ────────────────────────────────────────────────────────────────────────────────
def batched_fill_and_filter_nouns(
    fill,
    masked_texts: List[str],
    k: int,
    mask_token: str,
    batch_size: int = 64,
    seq_len: Optional[int] = None,
) -> List[List[str]]:
    """list 입력 → list[list[pred_token]] 반환 (명사 필터링, 길이 안전)"""
    clipped = [
        clip_around_mask(t, mask_token, seq_len=seq_len)
        for t in masked_texts
    ]
    outs = fill(clipped, top_k=max(50, k), batch_size=batch_size)
    preds_all: List[List[str]] = []
    # 파이프라인은 list 입력 시 list[list[dict]]를 반환
    for item in outs:
        cand: List[str] = []
        for p in item:
            token_str = p.get("token_str", "").strip().replace("##", "")
            if token_str and is_noun(token_str):
                cand.append(token_str)
            if len(cand) >= k:
                break
        preds_all.append(cand)
    return preds_all


def eval_lambada_topk(
    fill,
    eval_items: List[Dict[str, str]],
    mask_token: str,
    k: int = 5,
    batch_size: int = 64,
    seq_len: Optional[int] = None,
) -> float:
    masked = [it["masked"] for it in eval_items]
    golds = [it["target"] for it in eval_items]
    preds = batched_fill_and_filter_nouns(
        fill,
        masked,
        k=k,
        mask_token=mask_token,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    ok = 0
    for g, cands in zip(golds, preds):
        if not cands:
            continue
        # 정확히 일치 or 포함관계(어절/서브워드) 허용
        hit = (g in cands) or any((g in c) or (c in g) for c in cands)
        if hit:
            ok += 1
    return ok / max(1, len(golds))


def eval_coref_recall_topk(
    fill,
    eval_items: List[Dict[str, Any]],
    mask_token: str,
    k: int = 5,
    batch_size: int = 64,
    seq_len: Optional[int] = None,
) -> float:
    masked = [it["masked"] for it in eval_items]
    ctx_nouns = [set(it["context_nouns"]) for it in eval_items]
    preds = batched_fill_and_filter_nouns(
        fill,
        masked,
        k=k,
        mask_token=mask_token,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    ok = 0
    for ctx, cands in zip(ctx_nouns, preds):
        if any(c in ctx for c in cands):
            ok += 1
    return ok / max(1, len(eval_items))


def eval_coref_f1(
    fill,
    eval_items: List[Dict[str, Any]],
    mask_token: str,
    k: int = 5,
    batch_size: int = 64,
    seq_len: Optional[int] = None,
) -> float:
    """Coref F1 Score 계산"""
    masked = [it["masked"] for it in eval_items]
    ctx_nouns = [set(it["context_nouns"]) for it in eval_items]
    preds = batched_fill_and_filter_nouns(
        fill,
        masked,
        k=k,
        mask_token=mask_token,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    tp = fp = fn = 0
    for ctx, cands in zip(ctx_nouns, preds):
        pred_set = set(cands)
        ctx_set = ctx

        # True Positives: 예측된 명사가 문맥에 있음
        tp += len(pred_set & ctx_set)
        # False Positives: 예측된 명사가 문맥에 없음
        fp += len(pred_set - ctx_set)
        # False Negatives: 문맥에 있는 명사를 예측하지 못함
        fn += len(ctx_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


# ────────────────────────────────────────────────────────────────────────────────
# 메모리 한도 내 최대 per-device BS 탐색 + grad_acc 산정(throughput 극대화)
# ────────────────────────────────────────────────────────────────────────────────
def _sample_texts_for_probe(num=256) -> List[str]:
    try:
        ds = load_ko_lambada_split(prefer=("validation", "dev", "test"))
        out = []
        for i in range(min(num, len(ds))):
            ex = ds[i]
            t = ex.get("text") or ex.get("context") or ""
            target = ex.get("target") or ex.get("answer") or ""
            if t and target:
                out.append((t + " ").strip() + MASK_TOKEN_FALLBACK)
        return out if out else ["한국어 문맥에서 [MASK]을(를) 예측합니다."]
    except Exception:
        return ["한국어 문맥에서 [MASK]을(를) 예측합니다."] * min(8, num)


def _try_one_step(model, tokenizer, texts, seq_len):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    ).to(device)
    labels = inputs["input_ids"].clone()
    probs = torch.rand_like(labels.float())
    labels[probs > 0.15] = -100
    labels[(labels == tokenizer.pad_token_id)] = -100
    out = model(**inputs, labels=labels)
    loss = out.loss
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()


def find_max_bs(model, tokenizer, seq_len, start=2, limit=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = _sample_texts_for_probe(num=512)
    bs = start
    last_ok = None

    def runner(n):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        _try_one_step(model, tokenizer, texts[:n], seq_len)

    # 지수 증가
    while bs <= limit:
        try:
            runner(bs)
            last_ok = bs
            bs *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            raise

    # 하향 선형 탐색
    if last_ok is None:
        bs = max(1, start // 2)
        while bs >= 1:
            try:
                runner(bs)
                last_ok = bs
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    bs //= 2
                else:
                    raise
    return last_ok or 1


# ────────────────────────────────────────────────────────────────────────────────
# Optuna Objective
# ────────────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial, model_name: str, planned_trials: int, train_limit: Optional[int] = None, seed: int = 42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(seed)

    # HPO 공간 (LR 범위 확대)
    lr = trial.suggest_float("learning_rate", 5e-6, 2e-4, log=True)
    wd = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    min_prob = trial.suggest_float("min_prob", 0.05, 0.15)
    max_prob = trial.suggest_float("max_prob", 0.20, 0.35)
    seq_len = trial.suggest_categorical("max_length", [256, 384, 512])
    # DeBERTa와 gradient checkpointing 호환성 문제로 비활성화
    grad_ckpt = False if "deberta" in model_name.lower() else trial.suggest_categorical("gradient_checkpointing", [False, True])
    llrd = trial.suggest_float("llrd", 0.85, 1.0)

    # 로드
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)
    if grad_ckpt:
        mdl.gradient_checkpointing_enable()

    # 메모리 기반 BS/grad_acc 자동 산정
    max_bs = find_max_bs(mdl, tok, seq_len, start=2, limit=256)
    # 너무 작은 경우 학습 안정성을 위해 하한 2
    per_device_bs = max(2, max_bs)
    # throughput 극대화: grad_acc은 1로 시작(실제 step time은 콜백에서 관측)
    grad_acc = 1

    # 데이터/콜레이터
    tokenized = build_mlm_dataset(tok, max_length=seq_len, split="train", limit=train_limit)
    collator = DynCollator(tokenizer=tok, mlm=True, min_prob=min_prob, max_prob=max_prob, max_length=seq_len)

    # 총 스텝(대시보드 ETA 표시용)
    epochs = 1
    total_steps = math.ceil(len(tokenized) / (per_device_bs) / max(1, grad_acc)) * epochs

    # 옵티마이저(Layer-wise LR Decay)
    no_decay = ["bias", "LayerNorm.weight"]
    named = list(mdl.named_parameters())
    total_layers = sum(1 for n, _ in named if "encoder.layer" in n) or 12
    groups = []
    for n, p in named:
        lr_here = lr
        if "encoder.layer." in n:
            try:
                k = int(n.split("encoder.layer.")[1].split(".")[0])
            except Exception:
                k = total_layers - 1
            lr_here = lr * (llrd ** (total_layers - 1 - k))
        wd_here = 0.0 if any(nd in n for nd in no_decay) else wd
        groups.append({"params": [p], "weight_decay": wd_here, "lr": lr_here})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)

    # HP/데이터셋 메타 (대시보드 표출용)
    hp = {
        "seq_len": seq_len,
        "per_device_bs": per_device_bs,
        "grad_acc": grad_acc,
        "effective_tokens_per_update": per_device_bs * grad_acc * seq_len,
        "warmup_ratio": warmup,
        "lr": lr,
        "weight_decay": wd,
        "min_prob": min_prob,
        "max_prob": max_prob,
        "bf16": torch.cuda.is_available(),
        "train_limit": train_limit,
    }
    dataset_meta = {"corpus": "wikimedia/wikipedia", "subset": "20231101.ko", "split": "train"}

    # 학습 시작 이벤트 (Study ETA 계산을 위해)
    BUS.log(event="trial_begin", model=model_name, trial=trial.number, ts=time.time(), study_trials_total=planned_trials)

    args = TrainingArguments(
        output_dir=f"./runs/{model_name.replace('/','_')}/{trial.number}",
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=max(2, per_device_bs),
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        weight_decay=wd,
        warmup_ratio=warmup,
        num_train_epochs=epochs,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=seed,
        dataloader_drop_last=False,
        bf16=torch.cuda.is_available(),
    )

    tr = Trainer(
        model=mdl,
        args=args,
        train_dataset=tokenized,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tok,
        optimizers=(opt, None),
    )

    # 라이브 콜백(스텝 시간/처리량/손실/HP/ETA)
    tr.add_callback(
        LiveMetricsCallback(
            model_name=model_name,
            trial_id=trial.number,
            hp=hp,
            dataset_meta=dataset_meta,
            total_steps=total_steps,
            emit_every_steps=10,
        )
    )

    # 학습
    tr.train()

    # ===== 평가(실데이터 기반) =====
    # Fill-Mask 파이프라인 (주의: truncation/max_length kwargs 금지)
    fill = pipeline("fill-mask", model=mdl, tokenizer=tok, device=0 if device == "cuda" else -1)

    # Ko-LAMBADA 정확도
    eval_lbd = build_eval_from_lambada(limit=600)
    l_t1 = eval_lambada_topk(
        fill,
        eval_lbd,
        mask_token=tok.mask_token or MASK_TOKEN_FALLBACK,
        k=1,
        batch_size=64,
        seq_len=seq_len,
    )

    # Coref 평가 (F1 + top5 유지)
    eval_coref = build_coref_eval_set(limit=1600, max_seq_len=seq_len)  # 규모 증가
    c_f1 = eval_coref_f1(
        fill,
        eval_coref,
        mask_token=tok.mask_token or MASK_TOKEN_FALLBACK,
        k=5,
        batch_size=64,
        seq_len=seq_len,
    )
    c_t5 = eval_coref_recall_topk(
        fill,
        eval_coref,
        mask_token=tok.mask_token or MASK_TOKEN_FALLBACK,
        k=5,
        batch_size=64,
        seq_len=seq_len,
    )

    # 대시보드 송신
    BUS.log(section="eval_stream", model=model_name, trial=trial.number, lbd_top1=l_t1, coref_f1=c_f1, coref_top5=c_t5)

    # 스코어(F1 + top5 기반)
    score = 0.4 * c_f1 + 0.3 * c_t5 + 0.3 * l_t1
    trial.set_user_attr("lbd_top1", l_t1)
    trial.set_user_attr("coref_f1", c_f1)
    trial.set_user_attr("coref_top5", c_t5)

    # 학습 종료 이벤트
    BUS.log(event="trial_end", model=model_name, trial=trial.number, ts=time.time())

    # 메모리 정리
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(score)


# ────────────────────────────────────────────────────────────────────────────────
# Study 실행
# ────────────────────────────────────────────────────────────────────────────────
def run_study(model_name: str, n_trials: int = 15, seed: int = 42, train_limit: Optional[int] = None):
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=f"HPO-{model_name}")

    def _obj(t: optuna.Trial):
        return objective(t, model_name, planned_trials=n_trials, train_limit=train_limit, seed=seed)

    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)

    print("=== Best Trial ===")
    bt = study.best_trial
    print("score:", bt.value, "params:", bt.params, "attrs:", bt.user_attrs)
    return study


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Hugging Face model id")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-limit", type=int, default=60000)
    args = p.parse_args()

    run_study(args.model, n_trials=args.trials, seed=args.seed, train_limit=args.train_limit)
