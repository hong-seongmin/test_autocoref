#!/usr/bin/env python3
"""
체크포인트 재평가 스크립트 - Real Coref 메트릭 (대명사 없이 같은 명사 2번 이상)

새로운 평가 기준:
- Real@1: 2번째 명사를 마스킹했을 때 top-1 예측 정확도
- Real@5: 2번째 명사를 마스킹했을 때 top-5 예측 정확도
- LAMBADA@1: 기존 언어 모델링 성능 (유지)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# coref_automl 모듈에서 필요한 함수 import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from coref_automl.tune import (
    build_eval_from_lambada,
    build_real_coref_eval_set,
    eval_lambada_topk,
    eval_real_coref_top1,
    eval_real_coref_top5,
    detect_seq_len_from_checkpoint,
    MASK_TOKEN_FALLBACK,
)


def load_checkpoint_with_extended_embeddings(checkpoint_path: str):
    """
    확장된 sequence length를 가진 checkpoint 로드
    """
    from transformers import AutoConfig

    seq_len = detect_seq_len_from_checkpoint(checkpoint_path)

    print(f"   Loading checkpoint: {checkpoint_path}")
    if seq_len:
        print(f"   Detected seq_len: {seq_len}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if seq_len and seq_len > 512:
        # 확장된 임베딩 처리 - Config를 먼저 수정
        print(f"   Loading config and adjusting max_position_embeddings to {seq_len}...")
        config = AutoConfig.from_pretrained(checkpoint_path)
        original_max_pos = config.max_position_embeddings
        config.max_position_embeddings = seq_len
        print(f"   Config adjusted: {original_max_pos} → {seq_len}")

        # 수정된 config로 모델 로드
        print(f"   Loading model with adjusted config...")
        model = AutoModelForMaskedLM.from_pretrained(
            checkpoint_path,
            config=config,
            ignore_mismatched_sizes=True
        )

        # safetensors에서 rel_embeddings 수동 로드 (CPU에서)
        from safetensors import safe_open
        safetensors_path = Path(checkpoint_path) / "model.safetensors"

        if safetensors_path.exists():
            print(f"   Manually loading extended embeddings from safetensors...")
            with safe_open(str(safetensors_path), framework='pt', device='cpu') as f:
                # rel_embeddings 로드
                if 'deberta.encoder.rel_embeddings.weight' in f.keys():
                    rel_embed_weight = f.get_tensor('deberta.encoder.rel_embeddings.weight')
                    new_size, embed_dim = rel_embed_weight.shape
                    print(f"      rel_embeddings: loading [{new_size}, {embed_dim}]")

                    # 기존 embedding 확인
                    current_size = model.deberta.encoder.rel_embeddings.weight.shape[0]
                    print(f"      rel_embeddings: current size = {current_size}, target = {new_size}")

                    new_rel = torch.nn.Embedding(new_size, embed_dim)
                    new_rel.weight.data = rel_embed_weight.clone()
                    model.deberta.encoder.rel_embeddings = new_rel
                    print(f"      ✓ rel_embeddings loaded successfully: {model.deberta.encoder.rel_embeddings.weight.shape}")

                # position_embeddings 로드
                if 'deberta.embeddings.position_embeddings.weight' in f.keys():
                    pos_embed_weight = f.get_tensor('deberta.embeddings.position_embeddings.weight')
                    pos_size, pos_dim = pos_embed_weight.shape
                    print(f"      position_embeddings: loading [{pos_size}, {pos_dim}]")

                    current_pos_size = model.deberta.embeddings.position_embeddings.weight.shape[0]
                    print(f"      position_embeddings: current size = {current_pos_size}, target = {pos_size}")

                    new_pos = torch.nn.Embedding(pos_size, pos_dim)
                    new_pos.weight.data = pos_embed_weight.clone()
                    model.deberta.embeddings.position_embeddings = new_pos
                    print(f"      ✓ position_embeddings loaded successfully: {model.deberta.embeddings.position_embeddings.weight.shape}")
        else:
            print(f"   ⚠️  Warning: safetensors file not found, embeddings may not be correctly loaded")

        tokenizer.model_max_length = seq_len
        print(f"   ✓ Model loaded with seq_len={seq_len}")
    else:
        # 기본 모델 로드
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
        seq_len = model.config.max_position_embeddings
        print(f"   ✓ Model loaded with default seq_len={seq_len}")

    return model, tokenizer, seq_len


def evaluate_checkpoint(
    checkpoint_path: str,
    lambada_limit: int = 600,
    coref_limit: int = 1600,
) -> Dict[str, Any]:
    """
    단일 체크포인트 평가
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {checkpoint_path}")
    print(f"{'='*80}")

    start_time = time.time()

    # 모델 로드
    print("🔧 Loading model...")
    model, tokenizer, seq_len = load_checkpoint_with_extended_embeddings(checkpoint_path)
    device = 0 if torch.cuda.is_available() else -1
    model.eval()

    # Fill-mask 파이프라인 생성
    print("🔧 Creating fill-mask pipeline...")
    fill = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)
    mask_token = tokenizer.mask_token or MASK_TOKEN_FALLBACK

    # LAMBADA 평가
    print(f"\n📖 [1/2] Evaluating LAMBADA ({lambada_limit} samples)...")
    lambada_start = time.time()
    eval_lbd = build_eval_from_lambada(limit=lambada_limit)
    lambada_top1 = eval_lambada_topk(
        fill,
        eval_lbd,
        mask_token=mask_token,
        k=1,
        batch_size=64,
        seq_len=seq_len,
    )
    lambada_time = time.time() - lambada_start
    print(f"   ✓ LAMBADA@1 = {lambada_top1:.4f} ({lambada_time:.1f}s)")

    # Real Coref 평가
    print(f"\n🔗 [2/2] Building real coref evaluation set ({coref_limit} samples)...")
    coref_start = time.time()
    eval_coref = build_real_coref_eval_set(limit=coref_limit, max_seq_len=seq_len, seed=999)
    coref_samples = len(eval_coref)
    print(f"   ✓ Real coref set built: {coref_samples} samples")

    print("🔗 Evaluating Real@1...")
    real1_start = time.time()
    real1 = eval_real_coref_top1(
        fill,
        eval_coref,
        mask_token=mask_token,
        batch_size=64,
        seq_len=seq_len,
    )
    real1_time = time.time() - real1_start
    print(f"   ✓ Real@1 = {real1:.4f} ({real1_time:.1f}s)")

    print("🔗 Evaluating Real@5...")
    real5_start = time.time()
    real5 = eval_real_coref_top5(
        fill,
        eval_coref,
        mask_token=mask_token,
        batch_size=64,
        seq_len=seq_len,
    )
    real5_time = time.time() - real5_start
    print(f"   ✓ Real@5 = {real5:.4f} ({real5_time:.1f}s)")

    total_time = time.time() - start_time

    # 종합 점수 계산
    score = 0.4 * real1 + 0.3 * real5 + 0.3 * lambada_top1

    print(f"\n{'─'*80}")
    print(f"✅ Evaluation complete!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Score: {score:.4f}")
    print(f"{'─'*80}\n")

    # 체크포인트 정보 파싱
    checkpoint_name = Path(checkpoint_path).name

    # trainer_state.json에서 epoch 정보 읽기
    trainer_state_path = Path(checkpoint_path) / "trainer_state.json"
    epoch = None
    global_step = None

    if trainer_state_path.exists():
        try:
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
                epoch = trainer_state.get("epoch")
                global_step = trainer_state.get("global_step")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not read trainer_state.json: {e}")

    # checkpoint 이름에서 global_step 추출 (backup)
    if global_step is None:
        try:
            if "checkpoint-" in checkpoint_name:
                global_step = int(checkpoint_name.split("checkpoint-")[1])
        except Exception:
            pass

    result = {
        "checkpoint": checkpoint_path,
        "seq_len": seq_len,
        "evaluated_at": datetime.now().isoformat(),
        "lambada_top1": lambada_top1,
        "lambada_time": lambada_time,
        "coref_samples": coref_samples,
        "real1": real1,
        "real1_time": real1_time,
        "real5": real5,
        "real5_time": real5_time,
        "total_time": total_time,
        "score": score,
        "checkpoint_info": {
            "path": checkpoint_path,
            "name": checkpoint_name,
            "seq_len": seq_len,
            "model_type": model.config.model_type,
            "global_step": global_step,
            "epoch": epoch,
        }
    }

    # 메모리 정리
    del model
    del fill
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def find_checkpoints(base_dir: str) -> List[str]:
    """
    base_dir에서 모든 checkpoint 디렉토리 찾기
    """
    base_path = Path(base_dir)
    checkpoints = []

    # checkpoint-* 패턴 찾기
    for checkpoint_dir in base_path.rglob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            # config.json이 있는지 확인
            if (checkpoint_dir / "config.json").exists():
                checkpoints.append(str(checkpoint_dir))

    # 정렬 (checkpoint 번호 순)
    checkpoints.sort(key=lambda x: int(Path(x).name.split("checkpoint-")[1]) if "checkpoint-" in x else 0)

    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate checkpoints with Real Coref metrics")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="runs/combined_experiment",
        help="Base directory containing checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Single checkpoint path to evaluate"
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
        help="Number of coref samples"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="real_coref_results.json",
        help="Output JSON file path"
    )

    args = parser.parse_args()

    # 체크포인트 목록 결정
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        print(f"🔍 Searching for checkpoints in: {args.checkpoint_dir}")
        checkpoints = find_checkpoints(args.checkpoint_dir)
        print(f"   Found {len(checkpoints)} checkpoints")

    if not checkpoints:
        print("❌ No checkpoints found!")
        return

    # 체크포인트 리스트 출력
    print(f"\n{'='*80}")
    print(f"📋 Checkpoints to evaluate ({len(checkpoints)} total):")
    print(f"{'='*80}")
    for i, cp in enumerate(checkpoints, 1):
        print(f"   {i}. {cp}")
    print(f"{'='*80}\n")

    # 평가 실행
    overall_start = time.time()
    all_results = []

    for i, checkpoint_path in enumerate(checkpoints, 1):
        print(f"\n{'#'*80}")
        print(f"# Progress: {i}/{len(checkpoints)} ({(i/len(checkpoints)*100):.1f}%)")
        print(f"{'#'*80}")

        try:
            result = evaluate_checkpoint(
                checkpoint_path,
                lambada_limit=args.lambada_limit,
                coref_limit=args.coref_limit,
            )
            all_results.append(result)

            # 중간 저장 (매 체크포인트마다)
            partial_output = Path(args.output).parent / "real_coref_results_partial.json"
            partial_output.parent.mkdir(parents=True, exist_ok=True)

            # 기존 파일이 있으면 읽어서 병합
            existing_results = []
            if partial_output.exists():
                try:
                    with open(partial_output, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, dict) and "results" in existing_data:
                            existing_results = existing_data["results"]
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not read existing partial results: {e}")

            # 중복 제거 (같은 checkpoint는 최신 결과만 유지)
            checkpoint_paths = {res["checkpoint"] for res in all_results}
            filtered_existing = [res for res in existing_results if res["checkpoint"] not in checkpoint_paths]
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
                json.dump(partial_summary, f, ensure_ascii=False, indent=2)

            print(f"   💾 Partial results saved to: {partial_output}")

        except Exception as e:
            print(f"   ❌ Error evaluating {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 최종 결과 저장
    total_elapsed = (time.time() - overall_start) / 60

    # 최고 성능 체크포인트 찾기
    best_by_score = max(all_results, key=lambda x: x["score"]) if all_results else None
    best_by_real1 = max(all_results, key=lambda x: x["real1"]) if all_results else None
    best_by_real5 = max(all_results, key=lambda x: x["real5"]) if all_results else None

    summary = {
        "evaluated_at": datetime.now().isoformat(),
        "total_checkpoints": len(checkpoints),
        "successful_evaluations": len(all_results),
        "elapsed_time_minutes": total_elapsed,
        "lambada_limit": args.lambada_limit,
        "coref_limit": args.coref_limit,
        "best_by_score": best_by_score,
        "best_by_real1": best_by_real1,
        "best_by_real5": best_by_real5,
        "results": all_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # CSV 저장
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("checkpoint,seq_len,lambada_top1,real1,real5,score,epoch\n")
        for r in all_results:
            epoch = r["checkpoint_info"].get("epoch", "")
            f.write(f"{r['checkpoint']},{r['seq_len']},{r['lambada_top1']:.4f},"
                   f"{r['real1']:.4f},{r['real5']:.4f},{r['score']:.4f},{epoch}\n")

    # 최종 요약 출력
    print(f"\n{'='*80}")
    print(f"🎉 All evaluations complete!")
    print(f"{'='*80}")
    print(f"   Total time: {total_elapsed:.1f} minutes")
    print(f"   Successful: {len(all_results)}/{len(checkpoints)} checkpoints")

    if best_by_score:
        print(f"\n   🏆 Best by Score:")
        print(f"      Checkpoint: {best_by_score['checkpoint']}")
        print(f"      Score: {best_by_score['score']:.4f}")
        print(f"      LAMBADA@1: {best_by_score['lambada_top1']:.4f}")
        print(f"      Real@1: {best_by_score['real1']:.4f}")
        print(f"      Real@5: {best_by_score['real5']:.4f}")

    print(f"\n   💾 Results saved to:")
    print(f"      JSON: {output_path}")
    print(f"      CSV:  {csv_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
