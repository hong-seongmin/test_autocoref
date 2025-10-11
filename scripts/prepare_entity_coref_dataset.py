#!/usr/bin/env python3
"""
Entity Replacement Coreference 데이터셋 생성
============================================

반복된 개체(entity)를 찾아서 두 번째 등장을 대명사로 치환하여
명확한 상호참조 학습 데이터 생성

전략:
1. 완전 반복 개체 찾기 (홍길동...홍길동)
2. 두 번째를 "그", "이", "그것" 등으로 치환
3. [MASK]로 학습 데이터 생성

데이터 소스:
- Wikipedia (반복 패턴 많음)
- KLUE MRC (개체 반복 많음)
- Naver News (인명/조직명 반복 많음)
"""

import os
import sys
import json
import gc
import random
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from kiwipiepy import Kiwi
from typing import Dict, Any, List, Tuple
from multiprocessing import Pool, cpu_count
import argparse
from datetime import datetime
import time

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# 반복 개체 찾기
# ============================================================================

def find_exact_repetitions(text: str, kiwi) -> Dict[str, List[int]]:
    """
    완전히 같은 고유명사가 2번 이상 나오는 경우만 찾기

    Returns:
        {"홍길동": [0, 45, 100], "삼성전자": [200, 350]}
    """
    try:
        tokens = kiwi.tokenize(text)
    except:
        return {}

    entity_positions = {}

    for token in tokens:
        # NNP (고유명사)만
        if token.tag == 'NNP' and len(token.form) >= 2:
            entity = token.form

            # 너무 일반적인 단어 제외
            if entity in ['것', '수', '등', '때', '곳', '점', '년', '월', '일']:
                continue

            if entity not in entity_positions:
                entity_positions[entity] = []
            entity_positions[entity].append(token.start)

    # 2번 이상 등장한 것만
    repeated = {e: pos for e, pos in entity_positions.items() if len(pos) >= 2}

    return repeated


def worker_find_repetitions(args):
    """멀티프로세싱용 워커"""
    text, idx = args

    # 프로세스별 Kiwi 인스턴스
    if not hasattr(worker_find_repetitions, '_kiwi'):
        worker_find_repetitions._kiwi = Kiwi()

    kiwi = worker_find_repetitions._kiwi

    repeated = find_exact_repetitions(text, kiwi)

    if repeated:
        return (idx, text, repeated)
    return None


# ============================================================================
# Coref 샘플 생성
# ============================================================================

def create_coref_examples(text: str, repeated_entities: Dict[str, List[int]]) -> List[Dict[str, Any]]:
    """
    반복 개체에서 Coref 학습 샘플 생성 (Option 2: 개체 직접 예측)

    전략:
    - 원본: "홍길동은 학생이다. 홍길동은 공부한다."
    - 생성: "홍길동은 학생이다. [MASK]는 공부한다."
    - 정답: "홍길동" (개체 자체)

    Args:
        text: 원본 텍스트
        repeated_entities: {"홍길동": [0, 45, 100]}

    Returns:
        [{
            "text": "홍길동은 학생이다. [MASK]는 공부한다.",
            "target": "홍길동",
            "antecedent_pos": 0,
            "coref_pos": 45,
            "distance": 45
        }, ...]
    """
    examples = []

    for entity, positions in repeated_entities.items():
        # 첫 번째는 선행사로 유지
        antecedent_pos = positions[0]

        # 2번째 이상 등장을 [MASK]로 치환 (최대 3개)
        for coref_pos in positions[1:4]:
            # 거리 계산
            distance = coref_pos - antecedent_pos

            # 너무 가까우면 제외 (10자 미만)
            if distance < 10:
                continue

            # 너무 멀면 제외 (2000자 초과)
            if distance > 2000:
                continue

            # [MASK] 생성
            # coref_pos의 entity를 [MASK]로 직접 치환
            text_before = text[:coref_pos]
            text_after = text[coref_pos + len(entity):]
            masked_text = text_before + "[MASK]" + text_after

            examples.append({
                "text": masked_text,
                "target": entity,  # ★ 핵심: 개체 자체가 정답
                "antecedent_pos": antecedent_pos,
                "coref_pos": coref_pos,
                "distance": distance
            })

    return examples


# ============================================================================
# 데이터셋별 처리
# ============================================================================

def process_wikipedia(max_samples: int = 100000, num_workers: int = 20) -> List[str]:
    """Wikipedia에서 반복 개체 있는 텍스트 수집"""

    print("\n" + "=" * 80)
    print(f"📊 Wikipedia 처리 중...")
    print("=" * 80)

    dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)

    texts_with_repetitions = []
    scanned = 0

    print(f"🔍 스캔 중 (목표: {max_samples:,}개 문서)...")

    start_time = time.time()

    for sample in dataset:
        scanned += 1
        text = sample['text']

        # 길이 필터
        if len(text) < 300 or len(text) > 3000:
            continue

        # \n\n로 단락 분리
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) < 300:
                continue

            texts_with_repetitions.append(para)

            if len(texts_with_repetitions) >= max_samples:
                break

        if len(texts_with_repetitions) >= max_samples:
            break

        # 진행 상황
        if scanned % 1000 == 0:
            elapsed = time.time() - start_time
            rate = scanned / elapsed
            remaining = (max_samples - len(texts_with_repetitions)) / rate if rate > 0 else 0
            print(f"  스캔: {scanned:,} 문서 | 수집: {len(texts_with_repetitions):,}/{max_samples:,} | "
                  f"속도: {rate:.1f} docs/s | 남은 시간: {remaining/60:.1f}분")

    print(f"✅ Wikipedia 수집 완료: {len(texts_with_repetitions):,}개 단락")
    return texts_with_repetitions


def process_klue_mrc(num_workers: int = 20) -> List[str]:
    """KLUE MRC에서 텍스트 수집"""

    print("\n" + "=" * 80)
    print(f"📊 KLUE MRC 처리 중...")
    print("=" * 80)

    dataset = load_dataset("klue", "mrc", split="train")
    print(f"✅ 원본 샘플 수: {len(dataset):,}")

    # context만 추출 (길이 필터)
    texts = [s['context'] for s in dataset if 300 <= len(s['context']) <= 3000]

    print(f"✅ KLUE MRC 수집 완료: {len(texts):,}개")
    return texts


def process_naver_news(num_workers: int = 20) -> List[str]:
    """Naver News에서 텍스트 수집"""

    print("\n" + "=" * 80)
    print(f"📊 Naver News 처리 중...")
    print("=" * 80)

    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")
    print(f"✅ 원본 샘플 수: {len(dataset):,}")

    # document만 추출 (길이 필터)
    texts = [s['document'] for s in dataset if 300 <= len(s['document']) <= 3000]

    print(f"✅ Naver News 수집 완료: {len(texts):,}개")
    return texts


# ============================================================================
# 메인 처리
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entity Replacement Coreference 데이터셋 생성")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[1536, 2048],
                        help="시퀀스 길이")
    parser.add_argument("--datasets", nargs="+",
                        choices=["wikipedia", "klue_mrc", "naver_news", "all"],
                        default=["all"], help="사용할 데이터셋")
    parser.add_argument("--output-dir", default="./prepared_datasets",
                        help="저장 디렉토리")
    parser.add_argument("--target-samples", type=int, default=40000,
                        help="목표 샘플 수 (기본: 40,000)")
    parser.add_argument("--wiki-docs", type=int, default=100000,
                        help="Wikipedia 문서 수 (기본: 100,000)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="워커 수")
    parser.add_argument("--model", default="kakaobank/kf-deberta-base",
                        help="토크나이저 모델")

    args = parser.parse_args()

    # 워커 수
    if args.num_workers is None:
        args.num_workers = max(4, cpu_count() - 4)

    print("\n" + "=" * 80)
    print("🚀 Entity Replacement Coreference 데이터셋 생성")
    print("=" * 80)
    print(f"목표 샘플 수: {args.target_samples:,}")
    print(f"시퀀스 길이: {args.seq_lengths}")
    print(f"워커 수: {args.num_workers}")
    print(f"데이터셋: {args.datasets}")

    overall_start = time.time()

    # 데이터셋 선택
    datasets_to_use = args.datasets
    if "all" in datasets_to_use:
        datasets_to_use = ["wikipedia", "klue_mrc", "naver_news"]

    # 1단계: 텍스트 수집
    print("\n" + "=" * 80)
    print("📥 1단계: 원본 텍스트 수집")
    print("=" * 80)

    all_texts = []

    if "wikipedia" in datasets_to_use:
        wiki_texts = process_wikipedia(args.wiki_docs, args.num_workers)
        all_texts.extend(wiki_texts)

    if "klue_mrc" in datasets_to_use:
        klue_texts = process_klue_mrc(args.num_workers)
        all_texts.extend(klue_texts)

    if "naver_news" in datasets_to_use:
        news_texts = process_naver_news(args.num_workers)
        all_texts.extend(news_texts)

    print(f"\n✅ 총 수집된 텍스트: {len(all_texts):,}개")

    # 2단계: 반복 개체 찾기 (병렬)
    print("\n" + "=" * 80)
    print(f"🔍 2단계: 반복 개체 찾기 (병렬 처리, {args.num_workers} 워커)")
    print("=" * 80)

    step2_start = time.time()

    # 병렬 처리
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(
            worker_find_repetitions,
            [(text, i) for i, text in enumerate(all_texts)],
            chunksize=100
        )

    # None 제거
    results = [r for r in results if r is not None]

    step2_elapsed = time.time() - step2_start

    print(f"✅ 반복 개체 발견: {len(results):,}/{len(all_texts):,} 텍스트 "
          f"({len(results)/len(all_texts)*100:.1f}%)")
    print(f"⏱️  처리 시간: {step2_elapsed:.1f}초")

    # 3단계: Coref 샘플 생성
    print("\n" + "=" * 80)
    print("🔨 3단계: Coreference 샘플 생성")
    print("=" * 80)

    step3_start = time.time()

    all_examples = []
    for idx, text, repeated in results:
        examples = create_coref_examples(text, repeated)
        all_examples.extend(examples)

        if len(all_examples) >= args.target_samples:
            break

    # 목표 샘플 수만큼 자르기
    all_examples = all_examples[:args.target_samples]

    step3_elapsed = time.time() - step3_start

    print(f"✅ 생성된 샘플: {len(all_examples):,}")
    print(f"⏱️  생성 시간: {step3_elapsed:.1f}초")

    # 통계
    print("\n📊 데이터셋 통계:")
    distances = [ex['distance'] for ex in all_examples]
    targets = [ex['target'] for ex in all_examples]

    import numpy as np
    print(f"  거리 (선행사 ↔ 상호참조):")
    print(f"    평균: {np.mean(distances):.1f} 문자")
    print(f"    중앙값: {np.median(distances):.1f} 문자")
    print(f"    범위: {np.min(distances)} ~ {np.max(distances)} 문자")

    print(f"\n  개체(target) 분포:")
    from collections import Counter
    target_counts = Counter(targets)
    print(f"    고유 개체 수: {len(target_counts):,}개")
    print(f"    Top 10 빈도:")
    for target, count in target_counts.most_common(10):
        print(f"      '{target}': {count:,}개 ({count/len(targets)*100:.1f}%)")

    # 개체 길이 분포
    entity_lengths = [len(t) for t in targets]
    print(f"\n  개체 길이 분포:")
    print(f"    평균: {np.mean(entity_lengths):.1f} 글자")
    print(f"    중앙값: {np.median(entity_lengths):.1f} 글자")
    print(f"    범위: {np.min(entity_lengths)} ~ {np.max(entity_lengths)} 글자")

    # 샘플 출력
    print(f"\n  샘플 예시:")
    for i, ex in enumerate(random.sample(all_examples, min(3, len(all_examples))), 1):
        preview = ex['text'][:100] + "..." if len(ex['text']) > 100 else ex['text']
        print(f"    {i}. 개체: '{ex['target']}' (거리: {ex['distance']} 문자)")
        print(f"       텍스트: {preview}")
        print()

    # 4단계: 시퀀스 길이별 저장
    print("\n" + "=" * 80)
    print("💾 4단계: 토큰화 및 저장")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_len in args.seq_lengths:
        print(f"\n📦 시퀀스 길이 {seq_len} 처리 중...")
        step4_start = time.time()

        # 텍스트만 추출
        texts = [ex['text'] for ex in all_examples]

        # Dataset 생성
        dataset = Dataset.from_dict({"text": texts})

        # 토큰화
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=2000,
            remove_columns=["text"],
            num_proc=args.num_workers,
            desc=f"토큰화 (seq_len={seq_len})"
        )

        # 저장
        save_path = output_dir / f"entity_coref_{seq_len}"
        save_path.mkdir(parents=True, exist_ok=True)
        tokenized.save_to_disk(str(save_path))

        step4_elapsed = time.time() - step4_start

        print(f"  ✅ 저장 완료: {save_path}")
        print(f"  📊 샘플 수: {len(tokenized):,}")
        print(f"  ⏱️  시간: {step4_elapsed:.1f}초")

        # 메타데이터 저장
        meta_path = save_path / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset_type": "entity_replacement_coreference",
                "num_samples": len(all_examples),
                "seq_len": seq_len,
                "created_at": datetime.now().isoformat(),
                "source_datasets": datasets_to_use,
                "num_unique_entities": len(target_counts),
                "top_entities": dict(target_counts.most_common(20)),
                "entity_length_stats": {
                    "mean": float(np.mean(entity_lengths)),
                    "median": float(np.median(entity_lengths)),
                    "min": int(np.min(entity_lengths)),
                    "max": int(np.max(entity_lengths))
                },
                "distance_stats": {
                    "mean": float(np.mean(distances)),
                    "median": float(np.median(distances)),
                    "min": int(np.min(distances)),
                    "max": int(np.max(distances))
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"  📝 메타데이터: {meta_path}")

    overall_elapsed = time.time() - overall_start

    # 최종 요약
    print("\n" + "=" * 80)
    print("✅ 데이터셋 생성 완료!")
    print("=" * 80)
    print(f"⏱️  전체 시간: {overall_elapsed:.1f}초 ({overall_elapsed/60:.1f}분)")
    print(f"📊 생성된 샘플: {len(all_examples):,}개")
    print(f"💾 저장 경로:")
    for seq_len in args.seq_lengths:
        print(f"  - {output_dir}/entity_coref_{seq_len}/")

    print("\n📖 다음 단계:")
    print("=" * 80)
    print("Fine-tuning 실행:")
    print()
    print("# seq_len=2048")
    print(f"uv run python scripts/run_entity_coref_finetune.py \\")
    print(f"  --checkpoint runs/combined_experiment/checkpoint-410 \\")
    print(f"  --dataset {output_dir}/entity_coref_2048 \\")
    print(f"  --seq-len 2048 \\")
    print(f"  --epochs 3")
    print()
    print("# seq_len=1536")
    print(f"uv run python scripts/run_entity_coref_finetune.py \\")
    print(f"  --checkpoint runs/combined_experiment/checkpoint-396 \\")
    print(f"  --dataset {output_dir}/entity_coref_1536 \\")
    print(f"  --seq-len 1536 \\")
    print(f"  --epochs 3")


if __name__ == "__main__":
    main()
