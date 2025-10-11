#!/usr/bin/env python3
"""
Entity Coreference Fine-tuning
===============================

기존 MLM 체크포인트에서 시작하여 Entity Replacement Coref 데이터로 추가 학습

특징:
- 체크포인트 자동 로드 (시퀀스 길이 보존)
- 실시간 진행 상황 표시
- ETA 계산
- 중간 평가 (Coref@5, Coref F1)
- 상세 로그
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
    TrainerCallback,
)
from transformers.trainer_callback import TrainerControl, TrainerState
import argparse

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# coref_automl 평가 함수 import
from coref_automl.tune import (
    build_eval_from_lambada as build_lambada_eval,
    build_real_coref_eval_set,
    eval_lambada_topk as eval_lambada,
    eval_real_coref_top1,
    eval_real_coref_top5,
)


# ============================================================================
# 체크포인트 감지
# ============================================================================

def detect_seq_len_from_checkpoint(checkpoint_path: str) -> Optional[int]:
    """체크포인트에서 시퀀스 길이 감지"""
    config_path = Path(checkpoint_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            seq_len = config.get("max_position_embeddings")
            if seq_len:
                print(f"✓ 감지된 seq_len={seq_len} from checkpoint")
                return int(seq_len)
        except Exception as e:
            print(f"⚠️  체크포인트 config 읽기 실패: {e}")

    # HuggingFace 모델
    try:
        config = AutoConfig.from_pretrained(checkpoint_path)
        seq_len = getattr(config, "max_position_embeddings", None)
        if seq_len:
            print(f"✓ 감지된 seq_len={seq_len} from model")
            return int(seq_len)
    except:
        pass

    return None


# ============================================================================
# 평가 함수들은 coref_automl.tune에서 import
# ============================================================================


# ============================================================================
# 실시간 콜백
# ============================================================================

class DetailedProgressCallback(TrainerCallback):
    """실시간 상황 표시 콜백 with Real@1/Real@5 평가"""

    def __init__(self, model, tokenizer, seq_len, output_dir):
        self.start_time = None
        self.step_times = []
        self.last_log_step = 0
        self.model = model
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.output_dir = output_dir
        self.eval_history = []  # 평가 결과 저장

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()
        print("\n" + "=" * 80)
        print("🚀 훈련 시작!")
        print("=" * 80)
        print(f"총 스텝 수: {state.max_steps}")
        print(f"배치 크기: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"평가 간격: {args.eval_steps} 스텝")
        num_evals = state.max_steps // args.eval_steps
        print(f"예상 평가 횟수: {num_evals}회")
        print("=" * 80 + "\n")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        current_time = time.time()
        elapsed = current_time - self.start_time

        # 10 스텝마다 상세 정보 출력
        if state.global_step % 10 == 0 and state.global_step != self.last_log_step:
            self.last_log_step = state.global_step

            # 평균 시간 계산
            if len(self.step_times) > 0:
                avg_step_time = sum(self.step_times[-50:]) / len(self.step_times[-50:])
            else:
                avg_step_time = elapsed / max(1, state.global_step)

            self.step_times.append(avg_step_time)

            # 진행률
            progress = state.global_step / state.max_steps * 100

            # ETA
            remaining_steps = state.max_steps - state.global_step
            eta_seconds = avg_step_time * remaining_steps
            eta = timedelta(seconds=int(eta_seconds))

            # 현재 loss
            loss = state.log_history[-1].get('loss', 0.0) if state.log_history else 0.0

            # 다음 평가까지 남은 스텝
            steps_to_next_eval = args.eval_steps - (state.global_step % args.eval_steps)
            if steps_to_next_eval == args.eval_steps:
                steps_to_next_eval = 0

            print(f"📊 [Step {state.global_step:4d}/{state.max_steps}] "
                  f"진행: {progress:5.1f}% | Loss: {loss:.4f} | "
                  f"속도: {1/avg_step_time:.1f} steps/s | ETA: {eta} | "
                  f"다음 평가: {steps_to_next_eval}스텝 후")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print("\n" + "=" * 80)
        print(f"📊 중간 평가 시작 (Step {state.global_step})")
        print("=" * 80)

        eval_start = time.time()

        # Fill-mask pipeline 생성
        device = 0 if torch.cuda.is_available() else -1
        fill = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer, device=device)
        mask_token = self.tokenizer.mask_token or "[MASK]"

        # LAMBADA 평가 (간단히 100개만)
        print("\n📖 [1/3] LAMBADA 평가 (100 샘플)...")
        lbd_eval = build_lambada_eval(limit=100, seed=42)
        lbd_t1 = eval_lambada(fill, lbd_eval, mask_token=mask_token, k=1, batch_size=64, seq_len=self.seq_len)
        print(f"   ✓ LAMBADA@1 = {lbd_t1:.4f} ({lbd_t1*100:.2f}%)")

        # Real Coref 평가 (200개 샘플)
        print("\n🔗 [2/3] Real Coref 세트 구축 (200 샘플)...")
        coref_limit = 200
        eval_coref = build_real_coref_eval_set(limit=coref_limit, seed=999, max_seq_len=self.seq_len)
        print(f"   ✓ {len(eval_coref)} 샘플 준비 완료")

        # Real@1
        print("\n🔗 [3a/3] Real@1 계산...")
        real1 = eval_real_coref_top1(fill, eval_coref, mask_token=mask_token, batch_size=64, seq_len=self.seq_len)
        print(f"   ✓ Real@1 = {real1:.4f} ({real1*100:.2f}%)")

        # Real@5
        print("\n🔗 [3b/3] Real@5 계산...")
        real5 = eval_real_coref_top5(fill, eval_coref, mask_token=mask_token, batch_size=64, seq_len=self.seq_len)
        print(f"   ✓ Real@5 = {real5:.4f} ({real5*100:.2f}%)")

        eval_elapsed = time.time() - eval_start

        # 결과 저장
        eval_result = {
            'step': state.global_step,
            'lambada_top1': lbd_t1,
            'real1': real1,
            'real5': real5,
            'time': eval_elapsed
        }
        self.eval_history.append(eval_result)

        # 결과 요약
        print("\n" + "─" * 80)
        print(f"✅ 평가 완료 (소요 시간: {eval_elapsed:.1f}초)")
        print("─" * 80)
        print(f"   LAMBADA@1: {lbd_t1*100:.2f}%")
        print(f"   Real@1:    {real1*100:.2f}%")
        print(f"   Real@5:    {real5*100:.2f}%")

        # 이전 평가와 비교
        if len(self.eval_history) > 1:
            prev = self.eval_history[-2]
            real1_delta = (real1 - prev['real1']) * 100
            real5_delta = (real5 - prev['real5']) * 100
            print(f"\n   변화 (이전 평가 대비):")
            print(f"   Real@1: {real1_delta:+.2f}%p")
            print(f"   Real@5: {real5_delta:+.2f}%p")

        print("─" * 80 + "\n")

        # 중간 평가 결과 저장
        eval_log_path = Path(self.output_dir) / "eval_history.json"
        with open(eval_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.eval_history, f, indent=2, ensure_ascii=False)
        print(f"💾 중간 평가 결과 저장: {eval_log_path}\n")

        # GPU 메모리 정리
        del fill
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and 'eval_loss' in logs:
            print(f"   Eval Loss: {logs.get('eval_loss', 0.0):.4f}")


# ============================================================================
# 메인
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entity Coreference Fine-tuning")
    parser.add_argument("--checkpoint", required=True,
                        help="시작 체크포인트 (예: runs/combined_experiment/checkpoint-410)")
    parser.add_argument("--dataset", required=True,
                        help="Coref 데이터셋 경로 (예: prepared_datasets/entity_coref_2048)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="시퀀스 길이 (기본: 체크포인트에서 자동 감지)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="에폭 수 (기본: 5)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="배치 크기 (기본: 16)")
    parser.add_argument("--gradient-accumulation", type=int, default=2,
                        help="Gradient accumulation steps (기본: 2)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="학습률 (기본: 2e-5)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup 비율 (기본: 0.1)")
    parser.add_argument("--eval-steps", type=int, default=200,
                        help="평가 간격 (기본: 200)")
    parser.add_argument("--output-dir", default="./runs/entity_coref_finetune",
                        help="출력 디렉토리")
    parser.add_argument("--run-name", default=None,
                        help="실행 이름 (기본: 자동 생성)")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 Entity Coreference Fine-tuning")
    print("=" * 80)
    print(f"체크포인트: {args.checkpoint}")
    print(f"데이터셋: {args.dataset}")
    print(f"에폭: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"학습률: {args.lr}")

    overall_start = time.time()

    # 1. 시퀀스 길이 감지
    if args.seq_len is None:
        args.seq_len = detect_seq_len_from_checkpoint(args.checkpoint)
        if args.seq_len is None:
            print("❌ 시퀀스 길이를 감지할 수 없습니다. --seq-len으로 명시해주세요.")
            sys.exit(1)

    print(f"✓ 시퀀스 길이: {args.seq_len}")

    # 2. 모델 및 토크나이저 로드
    print("\n📥 모델 로드 중...")
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # ★ DeBERTa 체크포인트인지 확인
    is_deberta_checkpoint = "deberta" in args.checkpoint.lower() or args.seq_len > 512

    if is_deberta_checkpoint and args.seq_len > 512:
        # DeBERTa with extended sequence length
        print(f"   체크포인트가 확장된 시퀀스 길이({args.seq_len})를 가지고 있습니다...")
        print(f"   ignore_mismatched_sizes=True로 로드합니다...")

        model = AutoModelForMaskedLM.from_pretrained(args.checkpoint, ignore_mismatched_sizes=True)

        # rel_embeddings 수동 로드
        from safetensors import safe_open
        checkpoint_path = Path(args.checkpoint) / "model.safetensors"
        if checkpoint_path.exists():
            with safe_open(str(checkpoint_path), framework='pt', device='cpu') as f:
                if 'deberta.encoder.rel_embeddings.weight' in f.keys():
                    rel_embed_weight = f.get_tensor('deberta.encoder.rel_embeddings.weight')
                    print(f"   rel_embeddings 수동 로드: {rel_embed_weight.shape}")

                    new_size, embed_dim = rel_embed_weight.shape
                    new_rel = torch.nn.Embedding(new_size, embed_dim)
                    new_rel.weight.data = rel_embed_weight.clone()
                    model.deberta.encoder.rel_embeddings = new_rel.to(model.device)

        model.config.max_position_embeddings = args.seq_len
        print(f"   ✓ 모델 max_position_embeddings: {model.config.max_position_embeddings}")
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.checkpoint)

    # 토크나이저 max_length 업데이트
    tokenizer.model_max_length = args.seq_len

    load_elapsed = time.time() - load_start
    print(f"✅ 모델 로드 완료 ({load_elapsed:.1f}초)")
    print(f"   모델 파라미터: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"   Tokenizer max_length: {tokenizer.model_max_length}")

    # 3. 데이터셋 로드
    print("\n📥 데이터셋 로드 중...")
    dataset_start = time.time()

    dataset = load_from_disk(args.dataset)

    # Train/eval split (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']

    dataset_elapsed = time.time() - dataset_start
    print(f"✅ 데이터셋 로드 완료 ({dataset_elapsed:.1f}초)")
    print(f"   훈련 샘플: {len(train_dataset):,}")
    print(f"   평가 샘플: {len(eval_dataset):,}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # 4. Training arguments
    if args.run_name is None:
        args.run_name = f"coref_finetune_{args.seq_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 총 스텝 계산
    total_steps = (len(train_dataset) // (args.batch_size * args.gradient_accumulation)) * args.epochs
    num_evals = total_steps // args.eval_steps

    print(f"\n📊 훈련 설정:")
    print(f"   총 스텝: {total_steps}")
    print(f"   평가 간격: {args.eval_steps} 스텝")
    print(f"   예상 평가 횟수: {num_evals}회")
    print(f"   Effective batch size: {args.batch_size * args.gradient_accumulation}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=4,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[DetailedProgressCallback(model, tokenizer, args.seq_len, output_dir)],
    )

    # 6. 훈련
    print("\n" + "=" * 80)
    print("🚀 Fine-tuning 시작!")
    print("=" * 80)

    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start

    print("\n" + "=" * 80)
    print(f"✅ Fine-tuning 완료! ({train_elapsed/60:.1f}분)")
    print("=" * 80)

    # 7. 최종 평가 (Real Coref metrics)
    print("\n" + "=" * 80)
    print("📊 최종 평가 시작...")
    print("=" * 80)
    print("✓ 평가 방식: Real Coref (Task-aligned)")
    print("  - 반복 명사 → 명사 자체 예측")
    print("  - 대명사 제외 (훈련 데이터와 동일한 패턴)")
    print("  - Wikipedia streaming (seed=999)")
    print("=" * 80)

    eval_start = time.time()

    device = 0 if torch.cuda.is_available() else -1
    fill = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)
    mask_token = tokenizer.mask_token or "[MASK]"

    # LAMBADA 평가
    print("\n📖 [1/3] LAMBADA 평가 (600 샘플)...")
    lbd_start = time.time()
    eval_lbd = build_lambada_eval(limit=600, seed=42)
    lbd_t1 = eval_lambada(
        fill,
        eval_lbd,
        mask_token=mask_token,
        k=1,
        batch_size=64,
        seq_len=args.seq_len,
    )
    lbd_elapsed = time.time() - lbd_start
    print(f"   ✓ LAMBADA@1 = {lbd_t1:.4f} ({lbd_elapsed:.1f}초)")

    # Real Coref 평가 세트 구축
    print("\n🔗 [2/3] Real Coref 평가 세트 구축...")
    print("   샘플 수: LAMBADA 600개 기준으로 계산된 coref 샘플")
    print("   데이터 소스: Wikipedia streaming (seed=999)")
    coref_build_start = time.time()

    # seq_len에 따라 coref_limit 조정
    coref_limit = 1600 if args.seq_len == 1536 else 1600
    eval_coref = build_real_coref_eval_set(
        limit=coref_limit,
        seed=999,  # 훈련 데이터와 다른 seed
        max_seq_len=args.seq_len
    )
    coref_build_elapsed = time.time() - coref_build_start
    actual_samples = len(eval_coref)
    print(f"   ✓ Real Coref 세트: {actual_samples} 샘플 ({coref_build_elapsed:.1f}초)")

    # Real@1 계산
    print("\n🔗 [3a/3] Real@1 계산...")
    real1_start = time.time()
    real1 = eval_real_coref_top1(
        fill,
        eval_coref,
        mask_token=mask_token,
        batch_size=64,
        seq_len=args.seq_len,
    )
    real1_elapsed = time.time() - real1_start
    print(f"   ✓ Real@1 = {real1:.4f} ({real1_elapsed:.1f}초)")

    # Real@5 계산
    print("\n🔗 [3b/3] Real@5 계산...")
    real5_start = time.time()
    real5 = eval_real_coref_top5(
        fill,
        eval_coref,
        mask_token=mask_token,
        batch_size=64,
        seq_len=args.seq_len,
    )
    real5_elapsed = time.time() - real5_start
    print(f"   ✓ Real@5 = {real5:.4f} ({real5_elapsed:.1f}초)")

    eval_elapsed = time.time() - eval_start

    # 8. 결과 저장
    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "lambada_top1": lbd_t1,
        "real1": real1,
        "real5": real5,
        "coref_samples": actual_samples,
        "train_time_minutes": train_elapsed / 60,
        "eval_time_seconds": eval_elapsed,
        "finished_at": datetime.now().isoformat()
    }

    results_path = output_dir / "final_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✅ 최종 결과")
    print("=" * 80)
    print(f"LAMBADA@1: {lbd_t1:.4f} ({lbd_t1*100:.2f}%)")
    print(f"Real@1:    {real1:.4f} ({real1*100:.2f}%)")
    print(f"Real@5:    {real5:.4f} ({real5*100:.2f}%)")
    print(f"Coref 샘플: {actual_samples}")
    print(f"\n훈련 시간: {train_elapsed/60:.1f}분")
    print(f"평가 시간: {eval_elapsed:.1f}초")
    print(f"\n저장 경로: {output_dir}")
    print(f"결과 파일: {results_path}")

    overall_elapsed = time.time() - overall_start
    print(f"\n⏱️  전체 시간: {overall_elapsed/60:.1f}분")

    # 비교 출력
    print("\n" + "=" * 80)
    print("📊 성능 비교")
    print("=" * 80)
    print("이전 베스트 (checkpoint-1600, seq_len=2048):")
    print("  - Real@1: 67.78%")
    print("  - Real@5: 82.44%")
    print(f"\n현재 모델 (seq_len={args.seq_len}):")
    print(f"  - Real@1: {real1*100:.2f}%")
    print(f"  - Real@5: {real5*100:.2f}%")

    # 개선률 계산
    prev_real1 = 0.6778
    prev_real5 = 0.8244
    real1_improvement = ((real1 - prev_real1) / prev_real1) * 100
    real5_improvement = ((real5 - prev_real5) / prev_real5) * 100

    print(f"\n변화:")
    print(f"  - Real@1: {real1_improvement:+.2f}%")
    print(f"  - Real@5: {real5_improvement:+.2f}%")


if __name__ == "__main__":
    main()
