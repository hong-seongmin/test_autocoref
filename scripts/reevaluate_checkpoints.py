#!/usr/bin/env python3
"""
Checkpoint Re-evaluation Script
================================

Combined_experiment 체크포인트들을 수정된 평가 로직으로 재평가

특징:
- 데이터 누수 제거 (Wikipedia streaming 사용, 다른 seed)
- LAMBADA@1, Coref F1, Coref@5 평가
- 결과를 JSON/CSV로 저장
- Before/After 비교
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from tqdm import tqdm

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from coref_automl.tune import (
    build_eval_from_lambada,
    build_coref_eval_set,
    eval_lambada_topk,
    eval_coref_f1,
    eval_coref_recall_topk,
)


def find_checkpoints(base_dir: str) -> List[str]:
    """체크포인트 디렉토리 찾기"""
    base_path = Path(base_dir)
    checkpoints = []

    # checkpoint-* 패턴 찾기
    for ckpt_dir in sorted(base_path.glob("checkpoint-*")):
        if ckpt_dir.is_dir() and (ckpt_dir / "config.json").exists():
            checkpoints.append(str(ckpt_dir))

    return checkpoints


def load_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """체크포인트 정보 로드"""
    config_path = Path(checkpoint_path) / "config.json"
    trainer_state_path = Path(checkpoint_path) / "trainer_state.json"

    info = {
        "path": checkpoint_path,
        "name": Path(checkpoint_path).name,
    }

    # Config 읽기
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        info["seq_len"] = config.get("max_position_embeddings", 512)
        info["model_type"] = config.get("model_type", "unknown")

    # Trainer state 읽기
    if trainer_state_path.exists():
        with open(trainer_state_path, encoding="utf-8") as f:
            state = json.load(f)
        info["global_step"] = state.get("global_step", 0)
        info["epoch"] = state.get("epoch", 0)

    return info


def evaluate_checkpoint(
    checkpoint_path: str,
    seq_len: int = 512,
    lambada_limit: int = 600,
    coref_limit: int = 1600,
) -> Dict[str, Any]:
    """단일 체크포인트 평가"""
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {Path(checkpoint_path).name}")
    print(f"{'='*80}")

    eval_start = time.time()

    # 모델 및 토크나이저 로드
    print("📥 Loading model and tokenizer...")
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # DeBERTa extended sequence length 처리
    is_deberta = "deberta" in checkpoint_path.lower() or seq_len > 512

    if is_deberta and seq_len > 512:
        print(f"   Loading DeBERTa with extended seq_len={seq_len}...")
        model = AutoModelForMaskedLM.from_pretrained(
            checkpoint_path,
            ignore_mismatched_sizes=True
        )

        # rel_embeddings 수동 로드
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
    print(f"   Building LAMBADA set...")
    lbd_start = time.time()
    eval_lbd = build_eval_from_lambada(limit=lambada_limit, seed=42)
    print(f"   Evaluating with fill-mask pipeline (batch_size=64)...")
    eval_start_time = time.time()
    lbd_t1 = eval_lambada_topk(
        fill,
        eval_lbd,
        mask_token=mask_token,
        k=1,
        batch_size=64,
        seq_len=seq_len,
    )
    lbd_elapsed = time.time() - lbd_start
    print(f"   ✓ LAMBADA@1 = {lbd_t1:.4f} ({lbd_elapsed:.1f}s total, {time.time()-eval_start_time:.1f}s inference)")
    results["lambada_top1"] = lbd_t1
    results["lambada_time"] = lbd_elapsed

    # Coref 평가 세트 구축 (데이터 누수 제거)
    print(f"\n🔗 [2/3] Building coref evaluation set ({coref_limit} samples)...")
    print("   Using streaming Wikipedia (seed=999, different from training) + KLUE validation")
    print("   This may take a few minutes...")
    coref_build_start = time.time()
    eval_coref = build_coref_eval_set(
        limit=coref_limit,
        seed=999,  # 훈련 seed와 다름 (123 → 999)
        max_seq_len=seq_len
    )
    coref_build_elapsed = time.time() - coref_build_start
    print(f"   ✓ Coref set built: {len(eval_coref)} samples ({coref_build_elapsed:.1f}s)")
    results["coref_samples"] = len(eval_coref)

    # Coref F1
    print(f"\n🔗 [3/3] Evaluating coref metrics...")
    print(f"   Computing Coref F1 on {len(eval_coref)} samples...")
    print(f"   (This involves predicting top-5 nouns for each masked pronoun)")
    coref_f1_start = time.time()
    c_f1 = eval_coref_f1(
        fill,
        eval_coref,
        mask_token=mask_token,
        k=5,
        batch_size=64,
        seq_len=seq_len,
    )
    coref_f1_elapsed = time.time() - coref_f1_start
    samples_per_sec = len(eval_coref) / coref_f1_elapsed
    print(f"   ✓ Coref F1 = {c_f1:.4f} ({coref_f1_elapsed:.1f}s, {samples_per_sec:.1f} samples/s)")
    results["coref_f1"] = c_f1
    results["coref_f1_time"] = coref_f1_elapsed

    # Coref@5
    print(f"   Computing Coref@5 on {len(eval_coref)} samples...")
    coref_t5_start = time.time()
    c_t5 = eval_coref_recall_topk(
        fill,
        eval_coref,
        mask_token=mask_token,
        k=5,
        batch_size=64,
        seq_len=seq_len,
    )
    coref_t5_elapsed = time.time() - coref_t5_start
    samples_per_sec = len(eval_coref) / coref_t5_elapsed
    print(f"   ✓ Coref@5 = {c_t5:.4f} ({coref_t5_elapsed:.1f}s, {samples_per_sec:.1f} samples/s)")
    results["coref_top5"] = c_t5
    results["coref_top5_time"] = coref_t5_elapsed

    eval_elapsed = time.time() - eval_start
    results["total_time"] = eval_elapsed

    # 종합 점수 (Combined_experiment와 동일한 가중치)
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


def load_original_results(checkpoint_path: str) -> Optional[Dict[str, float]]:
    """원본 평가 결과 로드 (있다면)"""
    # eval_2048.log, eval_1536.log 파일에서 결과 찾기
    base_dir = Path(checkpoint_path).parent
    checkpoint_name = Path(checkpoint_path).name

    for log_file in base_dir.glob("eval_*.log"):
        try:
            with open(log_file, encoding="utf-8") as f:
                content = f.read()

            # 해당 체크포인트의 결과 찾기
            if checkpoint_name in content:
                lines = content.split('\n')
                for line in lines:
                    if checkpoint_name in line:
                        # 예: "[1/5] Evaluating checkpoint-410... LAMBADA@1=0.3217 | Coref F1=0.0387 | Coref@5=0.7137"
                        import re
                        match = re.search(
                            r'LAMBADA@1=([\d.]+).*Coref F1=([\d.]+).*Coref@5=([\d.]+)',
                            line
                        )
                        if match:
                            return {
                                "lambada_top1": float(match.group(1)),
                                "coref_f1": float(match.group(2)),
                                "coref_top5": float(match.group(3)),
                            }
        except Exception:
            continue

    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Re-evaluate checkpoints")
    parser.add_argument(
        "--checkpoint-dir",
        default="/home/work/hongseongmin/corefer/runs/combined_experiment",
        help="Base directory containing checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        help="Specific checkpoint to evaluate (optional)"
    )
    parser.add_argument(
        "--output",
        default="./reevaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--lambada-limit",
        type=int,
        default=600,
        help="Number of LAMBADA samples"
    )
    parser.add_argument(
        "--coref-limit",
        type=int,
        default=1600,
        help="Number of Coref samples"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔄 Checkpoint Re-evaluation")
    print("="*80)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    overall_start = time.time()

    # 체크포인트 찾기
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = find_checkpoints(args.checkpoint_dir)

    if not checkpoints:
        print("❌ No checkpoints found!")
        return

    print(f"✓ Found {len(checkpoints)} checkpoint(s):\n")
    for ckpt in checkpoints:
        info = load_checkpoint_info(ckpt)
        print(f"   - {info['name']}: seq_len={info.get('seq_len', 'unknown')}, "
              f"step={info.get('global_step', 'unknown')}")
    print()

    # 평가 실행
    all_results = []

    # 평가 시간 추적
    checkpoint_times = []

    for i, ckpt in enumerate(checkpoints, 1):
        print(f"\n{'#'*80}")
        print(f"# [{i}/{len(checkpoints)}] Processing checkpoint")

        # 진행률 및 ETA 계산
        progress_pct = (i - 1) / len(checkpoints) * 100
        if checkpoint_times:
            avg_time = sum(checkpoint_times) / len(checkpoint_times)
            remaining = len(checkpoints) - i + 1
            eta_minutes = (avg_time * remaining) / 60
            print(f"# Progress: {progress_pct:.1f}% | Estimated remaining: {eta_minutes:.1f} min")
        else:
            print(f"# Progress: {progress_pct:.1f}% | Estimating...")

        print(f"{'#'*80}\n")

        info = load_checkpoint_info(ckpt)
        seq_len = info.get("seq_len", 512)

        # 재평가 (시간 측정)
        ckpt_start = time.time()
        results = evaluate_checkpoint(
            ckpt,
            seq_len=seq_len,
            lambada_limit=args.lambada_limit,
            coref_limit=args.coref_limit,
        )
        ckpt_elapsed = time.time() - ckpt_start
        checkpoint_times.append(ckpt_elapsed)

        # 원본 결과 로드 (비교용)
        original = load_original_results(ckpt)
        if original:
            results["original"] = original

            # 차이 계산
            print(f"\n📊 Comparison (Original → New):")
            print(f"   LAMBADA@1: {original['lambada_top1']:.4f} → {results['lambada_top1']:.4f} "
                  f"({results['lambada_top1'] - original['lambada_top1']:+.4f})")
            print(f"   Coref F1:  {original['coref_f1']:.4f} → {results['coref_f1']:.4f} "
                  f"({results['coref_f1'] - original['coref_f1']:+.4f})")
            print(f"   Coref@5:   {original['coref_top5']:.4f} → {results['coref_top5']:.4f} "
                  f"({results['coref_top5'] - original['coref_top5']:+.4f})")

        # 체크포인트 정보 추가
        results["checkpoint_info"] = info
        all_results.append(results)

        # 중간 저장 (매 체크포인트마다)
        partial_output = Path(args.output).parent / "reevaluation_results_partial.json"

        # 기존 파일이 있으면 읽어서 병합
        existing_results = []
        if partial_output.exists():
            try:
                with open(partial_output, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    # 리스트인 경우 (여러 실행 결과가 쌓인 경우)
                    if isinstance(existing_data, list):
                        for entry in existing_data:
                            if isinstance(entry, dict) and "results" in entry:
                                existing_results.extend(entry["results"])
                    # 단일 dict인 경우
                    elif isinstance(existing_data, dict) and "results" in existing_data:
                        existing_results = existing_data["results"]
            except Exception as e:
                print(f"   ⚠️  Warning: Could not read existing partial results: {e}")

        # 중복 제거 (같은 checkpoint는 최신 결과만 유지)
        checkpoint_paths = {res["checkpoint"] for res in all_results}
        filtered_existing = [res for res in existing_results if res["checkpoint"] not in checkpoint_paths]

        # 병합된 결과
        merged_results = filtered_existing + all_results

        partial_summary = {
            "evaluated_at": datetime.now().isoformat(),
            "status": "in_progress",
            "completed_checkpoints": i,
            "total_checkpoints": len(checkpoints),
            "progress_pct": (i / len(checkpoints)) * 100,
            "elapsed_time_minutes": (time.time() - overall_start) / 60,
            "lambada_limit": args.lambada_limit,
            "coref_limit": args.coref_limit,
            "total_evaluated": len(merged_results),
            "results": merged_results,
        }
        with open(partial_output, 'w', encoding='utf-8') as f:
            json.dump(partial_summary, f, indent=2, ensure_ascii=False)
        print(f"   💾 Partial results saved to: {partial_output} (total: {len(merged_results)} checkpoints)")

        # 중간 요약 출력
        print(f"\n{'─'*80}")
        print(f"📈 Current Progress: {i}/{len(checkpoints)} checkpoints completed")
        print(f"{'─'*80}")
        print(f"{'Checkpoint':<20} {'LAMBADA@1':>12} {'Coref F1':>12} {'Coref@5':>12} {'Score':>10}")
        print(f"{'─'*80}")
        for res in all_results:
            name = res["checkpoint_info"]["name"]
            print(f"{name:<20} {res['lambada_top1']:>12.4f} {res['coref_f1']:>12.4f} "
                  f"{res['coref_top5']:>12.4f} {res['score']:>10.4f}")
        print(f"{'─'*80}\n")

    overall_elapsed = time.time() - overall_start

    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "evaluated_at": datetime.now().isoformat(),
        "total_checkpoints": len(checkpoints),
        "total_time_minutes": overall_elapsed / 60,
        "lambada_limit": args.lambada_limit,
        "coref_limit": args.coref_limit,
        "results": all_results,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("✅ Re-evaluation Complete!")
    print(f"{'='*80}")
    print(f"Total time: {overall_elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")

    # 요약 테이블 출력
    print("Summary Table:")
    print("-" * 80)
    print(f"{'Checkpoint':<20} {'LAMBADA@1':>12} {'Coref F1':>12} {'Coref@5':>12} {'Score':>10}")
    print("-" * 80)
    for res in all_results:
        name = res["checkpoint_info"]["name"]
        print(f"{name:<20} {res['lambada_top1']:>12.4f} {res['coref_f1']:>12.4f} "
              f"{res['coref_top5']:>12.4f} {res['score']:>10.4f}")
    print("-" * 80)

    # CSV 저장
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("checkpoint,seq_len,global_step,lambada_top1,coref_f1,coref_top5,score\n")
        for res in all_results:
            info = res["checkpoint_info"]
            f.write(f"{info['name']},{info.get('seq_len', 'unknown')},{info.get('global_step', 'unknown')},"
                    f"{res['lambada_top1']:.4f},{res['coref_f1']:.4f},{res['coref_top5']:.4f},{res['score']:.4f}\n")

    print(f"\n✓ CSV saved to: {csv_path}\n")


if __name__ == "__main__":
    main()
