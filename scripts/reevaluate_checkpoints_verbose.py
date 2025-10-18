#!/usr/bin/env python3
"""
Verbose Checkpoint Re-evaluation Script
========================================

실시간 세부 진행 상황 표시
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from coref_automl.tune import build_eval_from_lambada, build_coref_eval_set
from coref_automl.coref_utils import is_noun


def clip_around_mask(text: str, mask_token: str, left_chars: int = 200, right_chars: int = 200, seq_len: Optional[int] = None) -> str:
    """마스크 주변 텍스트 클리핑"""
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
        return text


def batched_fill_and_filter_nouns_verbose(
    fill,
    masked_texts: List[str],
    k: int,
    mask_token: str,
    batch_size: int = 64,
    seq_len: Optional[int] = None,
    desc: str = "Processing"
) -> List[List[str]]:
    """배치 예측 (명사 필터링, 진행률 표시)"""
    clipped = [clip_around_mask(t, mask_token, seq_len=seq_len) for t in masked_texts]

    # 배치 처리
    preds_all = []
    num_batches = (len(clipped) + batch_size - 1) // batch_size

    with tqdm(total=len(clipped), desc=desc, unit="sample", ncols=100, leave=False) as pbar:
        for i in range(0, len(clipped), batch_size):
            batch = clipped[i:i+batch_size]
            outs = fill(batch, top_k=max(50, k), batch_size=len(batch))

            # 결과가 단일 아이템이면 리스트로 변환
            if not isinstance(outs[0], list):
                outs = [outs]

            for item in outs:
                cand = []
                for p in item:
                    token_str = p.get("token_str", "").strip().replace("##", "")
                    if token_str and is_noun(token_str):
                        cand.append(token_str)
                    if len(cand) >= k:
                        break
                preds_all.append(cand)

            pbar.update(len(batch))

    return preds_all


def eval_lambada_topk_verbose(fill, eval_items: List[Dict[str, str]], mask_token: str, k: int = 1, batch_size: int = 64, seq_len: Optional[int] = None) -> float:
    """LAMBADA 평가 (진행률 표시)"""
    print(f"   Evaluating {len(eval_items)} samples...")
    masked = [it["masked"] for it in eval_items]
    golds = [it["target"] for it in eval_items]

    preds = batched_fill_and_filter_nouns_verbose(
        fill, masked, k=k, mask_token=mask_token,
        batch_size=batch_size, seq_len=seq_len,
        desc="   LAMBADA"
    )

    ok = 0
    for g, cands in zip(golds, preds):
        if not cands:
            continue
        hit = (g in cands) or any((g in c) or (c in g) for c in cands)
        if hit:
            ok += 1

    return ok / max(1, len(golds))


def eval_coref_f1_verbose(fill, eval_items: List[Dict[str, Any]], mask_token: str, k: int = 5, batch_size: int = 64, seq_len: Optional[int] = None) -> float:
    """Coref F1 평가 (진행률 표시)"""
    print(f"   Evaluating {len(eval_items)} samples...")
    masked = [it["masked"] for it in eval_items]
    ctx_nouns = [set(it["context_nouns"]) for it in eval_items]

    preds = batched_fill_and_filter_nouns_verbose(
        fill, masked, k=k, mask_token=mask_token,
        batch_size=batch_size, seq_len=seq_len,
        desc="   Coref F1"
    )

    tp = fp = fn = 0
    for ctx, cands in zip(ctx_nouns, preds):
        pred_set = set(cands)
        ctx_set = ctx
        tp += len(pred_set & ctx_set)
        fp += len(pred_set - ctx_set)
        fn += len(ctx_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def eval_coref_recall_topk_verbose(fill, eval_items: List[Dict[str, Any]], mask_token: str, k: int = 5, batch_size: int = 64, seq_len: Optional[int] = None) -> float:
    """Coref@k 평가 (진행률 표시)"""
    print(f"   Evaluating {len(eval_items)} samples...")
    masked = [it["masked"] for it in eval_items]
    ctx_nouns = [set(it["context_nouns"]) for it in eval_items]

    preds = batched_fill_and_filter_nouns_verbose(
        fill, masked, k=k, mask_token=mask_token,
        batch_size=batch_size, seq_len=seq_len,
        desc="   Coref@5"
    )

    ok = 0
    for ctx, cands in zip(ctx_nouns, preds):
        if any(c in ctx for c in cands):
            ok += 1

    return ok / max(1, len(eval_items))


def evaluate_checkpoint_verbose(
    checkpoint_path: str,
    seq_len: int = 512,
    lambada_limit: int = 600,
    coref_limit: int = 1600,
) -> Dict[str, Any]:
    """단일 체크포인트 평가 (상세 진행률)"""
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {Path(checkpoint_path).name}")
    print(f"{'='*80}")

    eval_start = time.time()

    # 모델 로드
    print("📥 Loading model and tokenizer...")
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    is_deberta = "deberta" in checkpoint_path.lower() or seq_len > 512

    if is_deberta and seq_len > 512:
        print(f"   Loading DeBERTa with extended seq_len={seq_len}...")
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_path, ignore_mismatched_sizes=True)

        from safetensors import safe_open
        safetensors_path = Path(checkpoint_path) / "model.safetensors"
        if safetensors_path.exists():
            with safe_open(str(safetensors_path), framework='pt', device='cpu') as f:
                if 'deberta.encoder.rel_embeddings.weight' in f.keys():
                    rel_embed_weight = f.get_tensor('deberta.encoder.rel_embeddings.weight')
                    new_size, embed_dim = rel_embed_weight.shape
                    new_rel = torch.nn.Embedding(new_size, embed_dim)
                    new_rel.weight.data = rel_embed_weight.clone()
                    model.deberta.encoder.rel_embeddings = new_rel.to(model.device)

        model.config.max_position_embeddings = seq_len
    else:
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)

    tokenizer.model_max_length = seq_len

    load_elapsed = time.time() - load_start
    print(f"✅ Model loaded ({load_elapsed:.1f}s)")

    # Fill-mask pipeline
    device = 0 if torch.cuda.is_available() else -1
    fill = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)
    mask_token = tokenizer.mask_token or "[MASK]"

    results = {
        "checkpoint": checkpoint_path,
        "seq_len": seq_len,
        "evaluated_at": datetime.now().isoformat(),
    }

    # LAMBADA 평가
    print(f"\n📖 [1/3] LAMBADA evaluation ({lambada_limit} samples)...")
    lbd_start = time.time()
    eval_lbd = build_eval_from_lambada(limit=lambada_limit, seed=42)
    lbd_t1 = eval_lambada_topk_verbose(fill, eval_lbd, mask_token=mask_token, k=1, batch_size=64, seq_len=seq_len)
    lbd_elapsed = time.time() - lbd_start
    print(f"   ✓ LAMBADA@1 = {lbd_t1:.4f} ({lbd_elapsed:.1f}s)")
    results["lambada_top1"] = lbd_t1
    results["lambada_time"] = lbd_elapsed

    # Coref 평가 세트 구축
    print(f"\n🔗 [2/3] Building coref evaluation set ({coref_limit} samples)...")
    print("   Using streaming Wikipedia (seed=999) + KLUE validation")
    coref_build_start = time.time()

    # 진행률 표시하며 빌드
    with tqdm(total=coref_limit, desc="   Building", unit="sample", ncols=100) as pbar:
        eval_coref = []
        original_build = build_coref_eval_set

        # Wrapper to track progress
        import random
        from kiwipiepy import Kiwi
        from datasets import load_dataset
        from collections import defaultdict
        import math

        kiwi = Kiwi()
        rnd = random.Random(999)

        sources = [
            ("wikimedia/wikipedia", "20231101.ko", "train"),
            ("klue", "ynat", "validation"),
        ]

        items = []
        scale = max(1, seq_len // 512)
        effective_limit = max(coref_limit, int(coref_limit * scale))
        target_per_source = max(1, math.ceil(effective_limit / len(sources)))
        per_source_counts = defaultdict(int)

        for source, subset, split in sources:
            try:
                ds = load_dataset(source, subset, split=split, streaming=(source == "wikimedia/wikipedia"))

                if source == "wikimedia/wikipedia":
                    for sample in ds:
                        if per_source_counts[source] >= target_per_source:
                            break

                        text = sample.get('text', '')
                        if not text or len(text) < 50:
                            continue

                        # Coref 샘플 생성 로직 (간소화)
                        from coref_automl.tune import process_coref_text
                        limit_per_text = min(12, max(5, 5 + 2 * (scale - 1)))
                        window_radius = min(3, max(1, scale - 1))

                        try:
                            processed = process_coref_text(text, kiwi, limit_per_text=limit_per_text, window_radius=window_radius)
                            for item in processed:
                                if per_source_counts[source] >= target_per_source:
                                    break
                                if len(item.get("context_nouns", [])) >= 2:
                                    items.append(item)
                                    per_source_counts[source] += 1
                                    pbar.update(1)
                        except:
                            continue

                        if per_source_counts[source] >= target_per_source:
                            break
                else:
                    # KLUE 처리
                    idxs = list(range(len(ds)))
                    rnd.shuffle(idxs)
                    for i in idxs:
                        if per_source_counts[source] >= target_per_source:
                            break
                        text = f"{ds[i]['title']} {ds[i].get('content', '')}".strip()
                        if not text or len(text) < 50:
                            continue

                        from coref_automl.tune import process_coref_text
                        limit_per_text = min(12, max(5, 5 + 2 * (scale - 1)))
                        window_radius = min(3, max(1, scale - 1))

                        try:
                            processed = process_coref_text(text, kiwi, limit_per_text=limit_per_text, window_radius=window_radius)
                            for item in processed:
                                if per_source_counts[source] >= target_per_source:
                                    break
                                if len(item.get("context_nouns", [])) >= 2:
                                    items.append(item)
                                    per_source_counts[source] += 1
                                    pbar.update(1)
                        except:
                            continue
            except Exception as e:
                print(f"   Warning: {source} failed: {e}")
                continue

        eval_coref = items[:effective_limit]

    coref_build_elapsed = time.time() - coref_build_start
    print(f"   ✓ Coref set built: {len(eval_coref)} samples ({coref_build_elapsed:.1f}s)")
    results["coref_samples"] = len(eval_coref)

    # Coref F1
    print(f"\n🔗 [3/3] Evaluating coref metrics...")
    print("   Computing Coref F1...")
    coref_f1_start = time.time()
    c_f1 = eval_coref_f1_verbose(fill, eval_coref, mask_token=mask_token, k=5, batch_size=64, seq_len=seq_len)
    coref_f1_elapsed = time.time() - coref_f1_start
    print(f"   ✓ Coref F1 = {c_f1:.4f} ({coref_f1_elapsed:.1f}s)")
    results["coref_f1"] = c_f1
    results["coref_f1_time"] = coref_f1_elapsed

    # Coref@5
    print("   Computing Coref@5...")
    coref_t5_start = time.time()
    c_t5 = eval_coref_recall_topk_verbose(fill, eval_coref, mask_token=mask_token, k=5, batch_size=64, seq_len=seq_len)
    coref_t5_elapsed = time.time() - coref_t5_start
    print(f"   ✓ Coref@5 = {c_t5:.4f} ({coref_t5_elapsed:.1f}s)")
    results["coref_top5"] = c_t5
    results["coref_top5_time"] = coref_t5_elapsed

    eval_elapsed = time.time() - eval_start
    results["total_time"] = eval_elapsed

    score = 0.4 * c_f1 + 0.3 * c_t5 + 0.3 * lbd_t1
    results["score"] = score

    print(f"\n✅ Evaluation complete!")
    print(f"   Total time: {eval_elapsed:.1f}s")
    print(f"   Score: {score:.4f}")

    # 메모리 정리
    del model
    del fill
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    # 간단한 테스트
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()

    # seq_len 감지
    if args.seq_len is None:
        config_path = Path(args.checkpoint) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            args.seq_len = config.get("max_position_embeddings", 512)

    evaluate_checkpoint_verbose(args.checkpoint, seq_len=args.seq_len)
