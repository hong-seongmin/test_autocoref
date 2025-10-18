#!/usr/bin/env python3
"""
Checkpoint Evaluation Script
=============================

Standalone script to evaluate any checkpoint with Real@1, Real@5, and LAMBADA@1 metrics.

Usage:
    python scripts/evaluate_checkpoint.py --checkpoint runs/mlm_v2_scratch_1536/checkpoint-655
    python scripts/evaluate_checkpoint.py --checkpoint runs/mlm_v2_scratch_2048/checkpoint-440 --seq-len 2048
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, pipeline

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from coref_automl.tune import (
    build_eval_from_lambada,
    build_real_coref_eval_set,
    eval_lambada_topk,
    eval_real_coref_combined,
)


def detect_seq_len_from_checkpoint(checkpoint_path: str) -> Optional[int]:
    """체크포인트에서 시퀀스 길이 자동 감지"""
    config_path = Path(checkpoint_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            seq_len = config.get("max_position_embeddings")
            if seq_len:
                return int(seq_len)
        except Exception as e:
            print(f"⚠️  Config 읽기 실패: {e}")

    try:
        config = AutoConfig.from_pretrained(checkpoint_path)
        seq_len = getattr(config, "max_position_embeddings", None)
        if seq_len:
            return int(seq_len)
    except:
        pass

    return None


def load_checkpoint(checkpoint_path: str, seq_len: int):
    """체크포인트 로드 (DeBERTa 확장 임베딩 지원)"""
    print(f"\n📥 모델 로드 중: {checkpoint_path}")
    print(f"   시퀀스 길이: {seq_len}")

    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.model_max_length = seq_len

    # DeBERTa 체크포인트 처리
    is_deberta = "deberta" in checkpoint_path.lower() or seq_len > 512

    if is_deberta and seq_len > 512:
        print("   DeBERTa with extended embeddings detected...")
        print("   Loading with ignore_mismatched_sizes=True...")

        model = AutoModelForMaskedLM.from_pretrained(checkpoint_path, ignore_mismatched_sizes=True)

        # rel_embeddings 수동 로드
        from safetensors import safe_open
        checkpoint_file = Path(checkpoint_path) / "model.safetensors"
        if checkpoint_file.exists():
            with safe_open(str(checkpoint_file), framework='pt', device='cpu') as f:
                if 'deberta.encoder.rel_embeddings.weight' in f.keys():
                    rel_embed_weight = f.get_tensor('deberta.encoder.rel_embeddings.weight')
                    print(f"   rel_embeddings 수동 로드: {rel_embed_weight.shape}")

                    new_size, embed_dim = rel_embed_weight.shape
                    new_rel = torch.nn.Embedding(new_size, embed_dim)
                    new_rel.weight.data = rel_embed_weight.clone()
                    model.deberta.encoder.rel_embeddings = new_rel.to(model.device)

        model.config.max_position_embeddings = seq_len
    else:
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)

    load_elapsed = time.time() - load_start
    param_count = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"✅ 모델 로드 완료 ({load_elapsed:.1f}초)")
    print(f"   파라미터: {param_count:.1f}M")
    print(f"   Tokenizer max_length: {tokenizer.model_max_length}")

    return model, tokenizer


def run_evaluation(model, tokenizer, seq_len: int):
    """Real@1, Real@5, LAMBADA@1 평가 실행"""
    print("\n" + "=" * 80)
    print("📊 평가 시작")
    print("=" * 80)

    eval_start = time.time()

    # Fill-mask pipeline
    device = 0 if torch.cuda.is_available() else -1
    fill = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)
    mask_token = tokenizer.mask_token or "[MASK]"

    # 1. LAMBADA 평가
    print("\n📖 [1/3] LAMBADA 평가 (600 샘플)...")
    lbd_start = time.time()
    eval_lbd = build_eval_from_lambada(limit=600, seed=42)
    lbd_t1 = eval_lambada_topk(
        fill,
        eval_lbd,
        mask_token=mask_token,
        k=1,
        batch_size=64,
        seq_len=seq_len,
    )
    lbd_elapsed = time.time() - lbd_start
    print(f"   ✓ LAMBADA@1 = {lbd_t1:.4f} ({lbd_t1*100:.2f}%) - {lbd_elapsed:.1f}초")

    # 2. Real Coref 평가 세트 구축
    print("\n🔗 [2/3] Real Coref 평가 세트 구축 (1600 샘플)...")
    coref_build_start = time.time()
    eval_coref = build_real_coref_eval_set(
        limit=1600,
        seed=999,
        max_seq_len=seq_len
    )
    coref_build_elapsed = time.time() - coref_build_start
    actual_samples = len(eval_coref)
    print(f"   ✓ {actual_samples} 샘플 준비 완료 ({coref_build_elapsed:.1f}초)")

    # 3. Real@1 & Real@5 계산 (최적화: 한 번에 계산)
    print("\n🔗 [3/3] Real@1 & Real@5 계산 (최적화: 한 번의 추론으로 계산)...")
    real_start = time.time()
    real1, real5 = eval_real_coref_combined(
        fill,
        eval_coref,
        mask_token=mask_token,
        batch_size=64,
        seq_len=seq_len,
    )
    real_elapsed = time.time() - real_start
    print(f"   ✓ Real@1 = {real1:.4f} ({real1*100:.2f}%)")
    print(f"   ✓ Real@5 = {real5:.4f} ({real5*100:.2f}%)")
    print(f"   ✓ 소요 시간: {real_elapsed:.1f}초 (최적화로 약 50% 단축)")

    eval_elapsed = time.time() - eval_start

    # 결과 반환
    results = {
        "lambada_top1": lbd_t1,
        "real1": real1,
        "real5": real5,
        "coref_samples": actual_samples,
        "eval_time_seconds": eval_elapsed,
        "timings": {
            "lambada": lbd_elapsed,
            "coref_build": coref_build_elapsed,
            "real_combined": real_elapsed,  # Real@1 + Real@5 동시 계산
        }
    }

    return results


def print_results(checkpoint_path: str, seq_len: int, results: dict):
    """결과 출력"""
    print("\n" + "=" * 80)
    print("✅ 평가 결과")
    print("=" * 80)
    print(f"체크포인트: {checkpoint_path}")
    print(f"시퀀스 길이: {seq_len}")
    print(f"Coref 샘플: {results['coref_samples']}")
    print()
    print(f"LAMBADA@1: {results['lambada_top1']:.4f} ({results['lambada_top1']*100:.2f}%)")
    print(f"Real@1:    {results['real1']:.4f} ({results['real1']*100:.2f}%)")
    print(f"Real@5:    {results['real5']:.4f} ({results['real5']*100:.2f}%)")
    print()
    print(f"평가 시간: {results['eval_time_seconds']:.1f}초")

    # 이전 베스트와 비교
    print("\n" + "─" * 80)
    print("📊 성능 비교")
    print("─" * 80)
    print("이전 베스트 (checkpoint-1600, Entity+MLM, seq_len=2048):")
    print("  - Real@1: 67.78%")
    print("  - Real@5: 82.44%")
    print()
    print(f"현재 체크포인트 (seq_len={seq_len}):")
    print(f"  - Real@1: {results['real1']*100:.2f}%")
    print(f"  - Real@5: {results['real5']*100:.2f}%")

    # 변화율 계산
    prev_real1 = 0.6778
    prev_real5 = 0.8244
    real1_delta = (results['real1'] - prev_real1) * 100
    real5_delta = (results['real5'] - prev_real5) * 100

    print()
    print(f"변화:")
    print(f"  - Real@1: {real1_delta:+.2f}%p")
    print(f"  - Real@5: {real5_delta:+.2f}%p")

    # 종합 스코어 (0.4*Real@1 + 0.3*Real@5 + 0.3*LAMBADA@1)
    score = 0.4 * results['real1'] + 0.3 * results['real5'] + 0.3 * results['lambada_top1']
    prev_score = 0.4 * prev_real1 + 0.3 * prev_real5 + 0.3 * 0.6613  # 이전 베스트의 LAMBADA@1

    print()
    print(f"종합 스코어 (0.4*Real@1 + 0.3*Real@5 + 0.3*LAMBADA@1):")
    print(f"  - 이전: {prev_score:.4f}")
    print(f"  - 현재: {score:.4f}")
    print(f"  - 변화: {(score - prev_score)*100:+.2f}%p")
    print("=" * 80)


def save_results(checkpoint_path: str, seq_len: int, results: dict, output_dir: Optional[str] = None):
    """결과를 JSON 파일로 저장"""
    if output_dir is None:
        output_dir = Path(checkpoint_path).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(checkpoint_path).name
    output_file = output_dir / f"{checkpoint_name}_eval_results.json"

    full_results = {
        "checkpoint": str(checkpoint_path),
        "seq_len": seq_len,
        "evaluated_at": datetime.now().isoformat(),
        **results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 결과 저장: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint with Real@1, Real@5, and LAMBADA@1 metrics"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (e.g., runs/mlm_v2_scratch_1536/checkpoint-655)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length (default: auto-detect from checkpoint)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: same as checkpoint directory)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🔍 Checkpoint Evaluation")
    print("=" * 80)
    print(f"체크포인트: {args.checkpoint}")

    start_time = time.time()

    # 1. 시퀀스 길이 감지
    if args.seq_len is None:
        args.seq_len = detect_seq_len_from_checkpoint(args.checkpoint)
        if args.seq_len is None:
            print("❌ 시퀀스 길이를 감지할 수 없습니다. --seq-len으로 명시해주세요.")
            sys.exit(1)
        print(f"✓ 감지된 seq_len: {args.seq_len}")
    else:
        print(f"✓ 지정된 seq_len: {args.seq_len}")

    # 2. 모델 로드
    model, tokenizer = load_checkpoint(args.checkpoint, args.seq_len)

    # 3. 평가 실행
    results = run_evaluation(model, tokenizer, args.seq_len)

    # 4. 결과 출력
    print_results(args.checkpoint, args.seq_len, results)

    # 5. 결과 저장
    output_file = save_results(args.checkpoint, args.seq_len, results, args.output_dir)

    # 전체 시간
    total_time = time.time() - start_time
    print(f"\n⏱️  전체 소요 시간: {timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    main()
