# coref_automl/tune.py
from __future__ import annotations
import os
import gc
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import optuna
import torch
from datasets import load_dataset, disable_caching
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
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
# Checkpoint Sequence Length Detection
# ────────────────────────────────────────────────────────────────────────────────
def detect_seq_len_from_checkpoint(model_path: str) -> Optional[int]:
    """
    Detect the sequence length (max_position_embeddings) from a checkpoint.

    Args:
        model_path: Path to checkpoint directory or HuggingFace model ID

    Returns:
        Detected sequence length, or None if detection fails
    """
    # Try local checkpoint first
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            seq_len = config.get("max_position_embeddings")
            if seq_len:
                print(f"✓ Detected seq_len={seq_len} from local checkpoint: {model_path}")
                return int(seq_len)
        except Exception as e:
            print(f"Warning: Failed to read config from {config_path}: {e}")

    # Try HuggingFace model config
    try:
        config = AutoConfig.from_pretrained(model_path)
        seq_len = getattr(config, "max_position_embeddings", None)
        if seq_len:
            print(f"✓ Detected seq_len={seq_len} from HuggingFace model: {model_path}")
            return int(seq_len)
    except Exception as e:
        print(f"Info: Could not load HuggingFace config for {model_path}: {e}")

    return None


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
# Real Entity Coref 평가 셋
#  - 대명사 없이 같은 명사가 2번 이상 나오는 경우만 사용
#  - 2번째 명사를 [MASK]로 변환하고, 정답은 해당 명사
# ────────────────────────────────────────────────────────────────────────────────
def build_real_coref_eval_set(
    limit=2000,
    seed=123,
    max_seq_len: int = 512,
) -> List[Dict[str, Any]]:
    """
    대명사 없고 같은 명사가 2번 이상 나오는 실제 상호참조 평가 데이터 생성

    Returns:
        List of dicts with keys:
        - masked: 2번째 명사를 [MASK]로 바꾼 텍스트
        - target: 정답 명사
        - full_text: 원본 전체 텍스트
    """
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    rnd = random.Random(seed)

    # 다중 데이터 소스
    sources = [
        ("wikimedia/wikipedia", "20231101.ko", "train"),
        ("klue", "ynat", "validation"),
    ]

    PRONOUN_POS = {"NP"}
    NOUN_POS = {"NNG", "NNP"}

    items: List[Dict[str, Any]] = []
    scale = max(1, max_seq_len // 512)
    effective_limit = max(limit, int(limit * scale))
    target_per_source = max(1, math.ceil(effective_limit / len(sources)))

    for source, subset, split in sources:
        try:
            ds = load_dataset(source, subset, split=split, streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=10000)

            processed_count = 0
            for example in ds:
                if processed_count >= target_per_source:
                    break

                # 텍스트 추출
                if source == "wikimedia/wikipedia":
                    text = example["text"]
                elif source == "klue":
                    text = f"{example['title']} {example.get('content', '')}".strip()
                else:
                    continue

                if not text or len(text) < 100:
                    continue

                # 형태소 분석
                tokens = kiwi.tokenize(text)

                # 대명사가 있는지 확인
                has_pronoun = any(tk.tag in PRONOUN_POS for tk in tokens)
                if has_pronoun:
                    continue  # 대명사가 있으면 스킵

                # 명사 등장 위치 추적
                noun_positions = defaultdict(list)  # {명사: [(start, end, index), ...]}
                for idx, tk in enumerate(tokens):
                    if tk.tag in NOUN_POS and len(tk.form) >= 2:  # 2글자 이상 명사만
                        noun_positions[tk.form].append((tk.start, tk.end, idx))

                # 2번 이상 등장한 명사 찾기
                repeated_nouns = {noun: positions for noun, positions in noun_positions.items()
                                 if len(positions) >= 2}

                if not repeated_nouns:
                    continue

                # 각 반복 명사에 대해 샘플 생성 (텍스트당 최대 3개)
                samples_from_text = 0
                for noun, positions in repeated_nouns.items():
                    if samples_from_text >= 3:
                        break

                    # 2번째 등장 위치를 마스킹
                    second_pos = positions[1]
                    start, end, idx = second_pos

                    # [MASK] 생성
                    masked_text = text[:start] + MASK_TOKEN_FALLBACK + text[end:]

                    items.append({
                        "masked": masked_text,
                        "target": noun,
                        "full_text": text,
                    })

                    samples_from_text += 1
                    processed_count += 1

                    if processed_count >= target_per_source:
                        break

        except Exception as e:
            print(f"Warning: Failed to load {source}: {e}")
            continue

    # 셔플 및 제한
    rnd.shuffle(items)
    return items[:effective_limit]


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
    show_progress: bool = True,
) -> List[List[str]]:
    """list 입력 → list[list[pred_token]] 반환 (명사 필터링, 길이 안전)"""
    clipped = [
        clip_around_mask(t, mask_token, seq_len=seq_len)
        for t in masked_texts
    ]

    # 진행률 표시 추가
    if show_progress:
        try:
            from tqdm import tqdm
            num_batches = (len(clipped) + batch_size - 1) // batch_size
            print(f"      Processing {len(clipped)} samples in {num_batches} batches (batch_size={batch_size})...")

            preds_all: List[List[str]] = []
            with tqdm(total=len(clipped), desc="      Inference", unit="sample", ncols=100, leave=False) as pbar:
                for i in range(0, len(clipped), batch_size):
                    batch = clipped[i:i+batch_size]
                    outs = fill(batch, top_k=max(50, k), batch_size=len(batch))

                    # 결과 처리
                    if not isinstance(outs[0], list):
                        outs = [outs]

                    for item in outs:
                        cand: List[str] = []
                        for p in item:
                            token_str = p.get("token_str", "").strip().replace("##", "")
                            if token_str and is_noun(token_str):
                                cand.append(token_str)
                            if len(cand) >= k:
                                break
                        preds_all.append(cand)

                    pbar.update(len(batch))

            return preds_all
        except ImportError:
            # tqdm 없으면 기본 처리
            pass

    # 기본 처리 (진행률 없음)
    outs = fill(clipped, top_k=max(50, k), batch_size=batch_size)
    preds_all: List[List[str]] = []
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


def eval_real_coref_top1(
    fill,
    eval_items: List[Dict[str, Any]],
    mask_token: str,
    batch_size: int = 64,
    seq_len: Optional[int] = None,
) -> float:
    """
    Real Coref Top-1 정확도
    2번째 명사를 마스킹했을 때 top-1 예측이 정답 명사인지 확인
    """
    masked = [it["masked"] for it in eval_items]
    targets = [it["target"] for it in eval_items]
    preds = batched_fill_and_filter_nouns(
        fill,
        masked,
        k=1,
        mask_token=mask_token,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    ok = 0
    for target, cands in zip(targets, preds):
        if not cands:
            continue
        # Top-1 예측이 정답인지 확인 (정확히 일치 or 포함관계)
        hit = (target == cands[0]) or (target in cands[0]) or (cands[0] in target)
        if hit:
            ok += 1

    return ok / max(1, len(eval_items))


def eval_real_coref_top5(
    fill,
    eval_items: List[Dict[str, Any]],
    mask_token: str,
    batch_size: int = 64,
    seq_len: Optional[int] = None,
) -> float:
    """
    Real Coref Top-5 정확도
    2번째 명사를 마스킹했을 때 top-5 예측에 정답 명사가 포함되는지 확인
    """
    masked = [it["masked"] for it in eval_items]
    targets = [it["target"] for it in eval_items]
    preds = batched_fill_and_filter_nouns(
        fill,
        masked,
        k=5,
        mask_token=mask_token,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    ok = 0
    for target, cands in zip(targets, preds):
        if not cands:
            continue
        # Top-5 중에 정답이 있는지 확인 (정확히 일치 or 포함관계)
        hit = any((target == c) or (target in c) or (c in target) for c in cands)
        if hit:
            ok += 1

    return ok / max(1, len(eval_items))


def eval_real_coref_combined(
    fill,
    eval_items: List[Dict[str, Any]],
    mask_token: str,
    batch_size: int = 64,
    seq_len: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Real Coref Top-1과 Top-5를 한 번에 계산 (최적화)

    한 번의 모델 추론으로 top-5를 가져온 후 Real@1과 Real@5를 동시에 계산하여
    추론 시간을 약 50% 절감합니다.

    Returns:
        Tuple[real1, real5]: (Real@1 정확도, Real@5 정확도)
    """
    masked = [it["masked"] for it in eval_items]
    targets = [it["target"] for it in eval_items]

    # 한 번만 추론 (top-5)
    preds = batched_fill_and_filter_nouns(
        fill,
        masked,
        k=5,
        mask_token=mask_token,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    real1_ok = 0
    real5_ok = 0
    for target, cands in zip(targets, preds):
        if not cands:
            continue

        # Real@1: top-1만 확인
        if (target == cands[0]) or (target in cands[0]) or (cands[0] in target):
            real1_ok += 1

        # Real@5: top-5 중에 있는지 확인
        if any((target == c) or (target in c) or (c in target) for c in cands):
            real5_ok += 1

    total = max(1, len(eval_items))
    return real1_ok / total, real5_ok / total


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

    # Detect seq_len from checkpoint if available, otherwise use Optuna
    detected_seq_len = detect_seq_len_from_checkpoint(model_name)
    if detected_seq_len:
        seq_len = detected_seq_len
        print(f"→ Using detected seq_len={seq_len} from checkpoint")
    else:
        # Expanded choices to support various sequence lengths
        seq_len = trial.suggest_categorical("max_length", [256, 384, 512, 1536, 2048])
        print(f"→ Using Optuna-suggested seq_len={seq_len}")

    # DeBERTa와 gradient checkpointing 호환성 문제로 비활성화
    # Check both model_name and if it's a checkpoint (which might be DeBERTa)
    is_deberta = "deberta" in model_name.lower() or detected_seq_len is not None
    grad_ckpt = False if is_deberta else trial.suggest_categorical("gradient_checkpointing", [False, True])
    llrd = trial.suggest_float("llrd", 0.85, 1.0)

    # 로드 - For checkpoints with resized embeddings, handle specially
    tok = AutoTokenizer.from_pretrained(model_name)

    # Update tokenizer's model_max_length to match seq_len
    if detected_seq_len:
        tok.model_max_length = detected_seq_len
        print(f"→ Updated tokenizer model_max_length to {detected_seq_len}")

    if detected_seq_len:
        # Load model with ignore_mismatched_sizes to handle rel_embeddings size difference
        print(f"→ Loading checkpoint with ignore_mismatched_sizes=True...")
        mdl = AutoModelForMaskedLM.from_pretrained(model_name, ignore_mismatched_sizes=True)

        # Now manually load the resized rel_embeddings from the checkpoint
        from safetensors import safe_open
        checkpoint_path = Path(model_name) / "model.safetensors"
        if checkpoint_path.exists():
            with safe_open(str(checkpoint_path), framework='pt', device='cpu') as f:
                if 'deberta.encoder.rel_embeddings.weight' in f.keys():
                    rel_embed_weight = f.get_tensor('deberta.encoder.rel_embeddings.weight')
                    print(f"→ Manually loading rel_embeddings with shape {rel_embed_weight.shape}")

                    # Create new embedding with the checkpoint's size
                    new_size, embed_dim = rel_embed_weight.shape
                    new_rel = torch.nn.Embedding(new_size, embed_dim)
                    new_rel.weight.data = rel_embed_weight.clone()
                    mdl.deberta.encoder.rel_embeddings = new_rel.to(mdl.device)

                    # Also handle position_embeddings if present
                    if 'deberta.embeddings.position_embeddings.weight' in f.keys():
                        pos_embed_weight = f.get_tensor('deberta.embeddings.position_embeddings.weight')
                        print(f"→ Manually loading position_embeddings with shape {pos_embed_weight.shape}")
                        pos_size, pos_dim = pos_embed_weight.shape
                        new_pos = torch.nn.Embedding(pos_size, pos_dim)
                        new_pos.weight.data = pos_embed_weight.clone()
                        mdl.deberta.embeddings.position_embeddings = new_pos.to(mdl.device)

        mdl.config.max_position_embeddings = detected_seq_len
        print(f"✓ Loaded checkpoint with seq_len={detected_seq_len}")
    else:
        # Fresh model or HuggingFace model - load normally
        mdl = AutoModelForMaskedLM.from_pretrained(model_name)

        # Resize position embeddings if we need a different seq_len than the model default
        if seq_len > mdl.config.max_position_embeddings:
            print(f"→ Resizing position embeddings from {mdl.config.max_position_embeddings} to {seq_len}")

            # DebertaV2 uses relative position embeddings in encoder.rel_embeddings
            position_embed = getattr(mdl.deberta.embeddings, "position_embeddings", None)
            if position_embed is not None:
                old_num, dim = position_embed.weight.shape
                new_embed = torch.nn.Embedding(seq_len, dim)
                new_embed.weight.data[:old_num] = position_embed.weight.data.clone()
                if seq_len > old_num:
                    new_embed.weight.data[old_num:] = position_embed.weight.data[-1:].repeat(seq_len - old_num, 1)
                mdl.deberta.embeddings.position_embeddings = new_embed.to(mdl.device)

            rel_embeddings = getattr(mdl.deberta.encoder, "rel_embeddings", None)
            if rel_embeddings is not None:
                old_rel_num, rel_dim = rel_embeddings.weight.shape
                new_rel = torch.nn.Embedding(seq_len, rel_dim)
                new_rel.weight.data[:old_rel_num] = rel_embeddings.weight.data.clone()
                if seq_len > old_rel_num:
                    new_rel.weight.data[old_rel_num:] = rel_embeddings.weight.data[-1:].repeat(seq_len - old_rel_num, 1)
                mdl.deberta.encoder.rel_embeddings = new_rel.to(mdl.device)

            mdl.config.max_position_embeddings = seq_len
            print(f"✓ Position embeddings resized successfully")

    if grad_ckpt:
        mdl.gradient_checkpointing_enable()

    # 메모리 기반 BS/grad_acc 자동 산정
    max_bs = find_max_bs(mdl, tok, seq_len, start=2, limit=256)
    # 너무 작은 경우 학습 안정성을 위해 하한 2
    per_device_bs = max(1, max_bs // 2)  # 50% 마진
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
    print(f"\n{'─'*80}")
    print("📊 Starting evaluation phase...")
    print(f"{'─'*80}\n")

    # Fill-Mask 파이프라인 (주의: truncation/max_length kwargs 금지)
    print("🔧 Creating fill-mask pipeline...")
    fill = pipeline("fill-mask", model=mdl, tokenizer=tok, device=0 if device == "cuda" else -1)

    # Ko-LAMBADA 정확도
    print("📖 [1/2] Evaluating LAMBADA (600 samples)...")
    eval_lbd = build_eval_from_lambada(limit=600)
    l_t1 = eval_lambada_topk(
        fill,
        eval_lbd,
        mask_token=tok.mask_token or MASK_TOKEN_FALLBACK,
        k=1,
        batch_size=64,
        seq_len=seq_len,
    )
    print(f"   ✓ LAMBADA@1 = {l_t1:.4f}")

    # Real Coref 평가 (대명사 없고 같은 명사 2번 이상)
    print("🔗 [2/2] Building real coref evaluation set (1600 samples)...")
    eval_coref = build_real_coref_eval_set(limit=1600, max_seq_len=seq_len, seed=999)
    print(f"   ✓ Real coref set built: {len(eval_coref)} samples")

    print("🔗 Evaluating real coref metrics...")
    real1 = eval_real_coref_top1(
        fill,
        eval_coref,
        mask_token=tok.mask_token or MASK_TOKEN_FALLBACK,
        batch_size=64,
        seq_len=seq_len,
    )
    print(f"   ✓ Real@1 = {real1:.4f}")

    real5 = eval_real_coref_top5(
        fill,
        eval_coref,
        mask_token=tok.mask_token or MASK_TOKEN_FALLBACK,
        batch_size=64,
        seq_len=seq_len,
    )
    print(f"   ✓ Real@5 = {real5:.4f}")

    print(f"\n{'─'*80}")
    print(f"✅ Evaluation complete!")
    print(f"{'─'*80}\n")

    # 대시보드 송신
    BUS.log(section="eval_stream", model=model_name, trial=trial.number, lbd_top1=l_t1, real1=real1, real5=real5)

    # 스코어 (Real1, Real5 기반)
    score = 0.4 * real1 + 0.3 * real5 + 0.3 * l_t1
    trial.set_user_attr("lbd_top1", l_t1)
    trial.set_user_attr("real1", real1)
    trial.set_user_attr("real5", real5)

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
class TeeLogger:
    """Duplicate stdout to both console and file"""
    def __init__(self, file_path):
        self.terminal = __import__('sys').stdout
        self.log = open(file_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def run_study(model_name: str, n_trials: int = 15, seed: int = 42, train_limit: Optional[int] = None):
    import sys
    from datetime import datetime, timedelta

    # Setup automatic logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("./runs") / "tune_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"tune_{model_name.replace('/', '_')}_{timestamp}.log"

    print(f"📝 Logging to: {log_file}")
    tee = TeeLogger(str(log_file))
    sys.stdout = tee

    # Track timing for ETA
    trial_times = []
    study_start_time = time.time()

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=f"HPO-{model_name}")

    def _obj(t: optuna.Trial):
        trial_start = time.time()

        # Print trial header
        avg_time = sum(trial_times) / len(trial_times) if trial_times else None
        if avg_time:
            est_trial_time = timedelta(seconds=int(avg_time))
            est_remaining = timedelta(seconds=int(avg_time * (n_trials - t.number)))
            print(f"\n{'='*80}")
            print(f"🚀 [Trial {t.number + 1}/{n_trials}] Starting...")
            print(f"   Estimated time per trial: {est_trial_time}")
            print(f"   Estimated remaining time: {est_remaining}")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"🚀 [Trial {t.number + 1}/{n_trials}] Starting (first trial, no ETA yet)...")
            print(f"{'='*80}\n")

        result = objective(t, model_name, planned_trials=n_trials, train_limit=train_limit, seed=seed)

        trial_elapsed = time.time() - trial_start
        trial_times.append(trial_elapsed)

        # Print trial summary
        elapsed_str = str(timedelta(seconds=int(trial_elapsed)))

        # Get best score and attrs safely
        try:
            best_score = study.best_value
            best_attrs = study.best_trial.user_attrs
        except (ValueError, AttributeError):
            # First trial or no completed trials yet
            best_score = result
            best_attrs = t.user_attrs

        print(f"\n{'='*80}")
        print(f"✅ [Trial {t.number + 1}/{n_trials}] Completed in {elapsed_str}")
        print(f"   Score: {result:.4f}")
        print(f"   Metrics: LAMBADA@1={t.user_attrs.get('lbd_top1', 0):.4f} | "
              f"Real@1={t.user_attrs.get('real1', 0):.4f} | "
              f"Real@5={t.user_attrs.get('real5', 0):.4f}")
        print(f"   Best so far: {best_score:.4f} "
              f"(LAMBADA@1={best_attrs.get('lbd_top1', 0):.4f}, "
              f"Real@5={best_attrs.get('real5', 0):.4f})")

        # Overall progress
        completed = t.number + 1
        progress_pct = (completed / n_trials) * 100
        avg_time_per_trial = sum(trial_times) / len(trial_times)
        remaining_trials = n_trials - completed
        est_remaining = timedelta(seconds=int(avg_time_per_trial * remaining_trials))

        print(f"   Overall progress: {completed}/{n_trials} trials ({progress_pct:.1f}%) | "
              f"Estimated remaining: {est_remaining}")
        print(f"{'='*80}\n")

        return result

    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)

    total_elapsed = time.time() - study_start_time
    total_elapsed_str = str(timedelta(seconds=int(total_elapsed)))

    print(f"\n{'='*80}")
    print(f"🎉 Study completed in {total_elapsed_str}")
    print(f"{'='*80}")
    print("=== Best Trial ===")
    bt = study.best_trial
    print(f"Score: {bt.value:.4f}")
    print(f"Params: {bt.params}")
    print(f"Metrics: LAMBADA@1={bt.user_attrs.get('lbd_top1', 0):.4f} | "
          f"Real@1={bt.user_attrs.get('real1', 0):.4f} | "
          f"Real@5={bt.user_attrs.get('real5', 0):.4f}")
    print(f"{'='*80}\n")

    # Restore stdout and close log file
    sys.stdout = tee.terminal
    tee.close()
    print(f"✅ Log saved to: {log_file}")

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
