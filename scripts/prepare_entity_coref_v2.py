#!/usr/bin/env python3
"""
Entity Coreference Dataset V2 - High Quality News-based
=======================================================

고품질 뉴스 데이터로 Entity Coreference 데이터셋 생성 (개수 제한 없음)

데이터 소스 (개체 반복률 기준 선정):
1. dev7halo/naver-news-summarization-ko-with-gen (97.7% 반복률) ⭐⭐⭐⭐⭐
2. HPLT/HPLT2.0_cleaned kor_Hang (81.0% 반복률) ⭐⭐⭐⭐⭐
3. AIR-Bench/qa_news_ko (91.3% 반복률) ⭐⭐⭐⭐
4. nmixx-fin/twice_kr_finance_news_summ (90.3% 반복률) ⭐⭐⭐⭐
5. nmixx-fin/twice_kr_news_bqa_cls (89.7% 반복률) ⭐⭐⭐⭐

개선 사항:
- 개체 빈도 제한 (최대 1,000회 → 편향 방지)
- 대명사 필터링 (NP 태그 제외)
- 개수 제한 없음 (모든 고품질 데이터 활용)
- 거리 다양화 (10-2000자)
"""

import os
import sys
import json
import gc
import random
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
from multiprocessing import Pool, cpu_count
import argparse
from datetime import datetime

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from kiwipiepy import Kiwi
import numpy as np

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# 반복 개체 찾기
# ============================================================================

def find_exact_repetitions(text: str, kiwi) -> Dict[str, List[int]]:
    """
    완전히 같은 고유명사가 2번 이상 나오는 경우만 찾기 (대명사 제외)

    Returns:
        {"홍길동": [0, 45, 100], "삼성전자": [200, 350]}
    """
    try:
        tokens = kiwi.tokenize(text)
    except:
        return {}

    # 대명사 체크 (강화)
    PRONOUN_POS = {"NP"}
    has_pronoun = any(tk.tag in PRONOUN_POS for tk in tokens)
    if has_pronoun:
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
# 병렬 품질 필터링
# ============================================================================

def quality_check_worker(text: str):
    """병렬 처리용 품질 체크 워커"""
    # 프로세스별 Kiwi 인스턴스 (재사용)
    if not hasattr(quality_check_worker, '_kiwi'):
        quality_check_worker._kiwi = Kiwi()

    kiwi = quality_check_worker._kiwi

    try:
        # NNP 비율 체크
        tokens = kiwi.tokenize(text)
        nnp_count = sum(1 for t in tokens if t.tag == 'NNP')
        nnp_ratio = nnp_count / max(1, len(tokens))

        if nnp_ratio < 0.05 or nnp_ratio > 0.20:
            return None

        # 문장 수 체크
        sentences = text.count('.') + text.count('?') + text.count('!')
        if sentences < 3:
            return None

        return text
    except:
        return None


# ============================================================================
# Coref 샘플 생성 (개체 빈도 제한 추가)
# ============================================================================

def create_coref_examples_with_limit(
    text: str,
    repeated_entities: Dict[str, List[int]],
    entity_counter: Counter,
    max_entity_freq: int = 1000
) -> List[Dict[str, Any]]:
    """
    반복 개체에서 Coref 학습 샘플 생성 (개체 빈도 제한)

    Args:
        text: 원본 텍스트
        repeated_entities: {"홍길동": [0, 45, 100]}
        entity_counter: 전체 개체 빈도 카운터
        max_entity_freq: 개체당 최대 샘플 수

    Returns:
        샘플 리스트
    """
    examples = []

    for entity, positions in repeated_entities.items():
        # 개체 빈도 제한 체크
        if entity_counter[entity] >= max_entity_freq:
            continue

        # 첫 번째는 선행사로 유지
        antecedent_pos = positions[0]

        # 2번째 이상 등장을 [MASK]로 치환 (최대 3개)
        for coref_pos in positions[1:4]:
            # 개체별 빈도 재체크
            if entity_counter[entity] >= max_entity_freq:
                break

            # 거리 계산
            distance = coref_pos - antecedent_pos

            # 너무 가까우면 제외 (10자 미만)
            if distance < 10:
                continue

            # 너무 멀면 제외 (2000자 초과)
            if distance > 2000:
                continue

            # [MASK] 생성
            text_before = text[:coref_pos]
            text_after = text[coref_pos + len(entity):]
            masked_text = text_before + "[MASK]" + text_after

            examples.append({
                "text": masked_text,
                "target": entity,
                "antecedent_pos": antecedent_pos,
                "coref_pos": coref_pos,
                "distance": distance
            })

            # 빈도 증가
            entity_counter[entity] += 1

    return examples


# ============================================================================
# 데이터셋별 처리
# ============================================================================

def process_naver_news_gen(num_workers: int = 20) -> List[str]:
    """dev7halo/naver-news-summarization-ko-with-gen (병렬 품질 필터)"""

    print("\n" + "=" * 80)
    print(f"📊 Naver News (최고 품질)")
    print("=" * 80)

    dataset = load_dataset("dev7halo/naver-news-summarization-ko-with-gen", split="train")
    print(f"✅ 원본 샘플 수: {len(dataset):,}")

    # 1단계: 길이 필터 (800-2500자)
    texts = [s['document'] for s in dataset if 800 <= len(s['document']) <= 2500]
    print(f"   길이 필터 후: {len(texts):,}개")

    # 2단계: 병렬 품질 필터
    print(f"   품질 필터링 중 ({num_workers} 워커)...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(quality_check_worker, texts, chunksize=100)

    filtered = [r for r in results if r is not None]

    print(f"✅ 수집 완료: {len(filtered):,}개")
    return filtered


def process_hplt_korean(num_workers: int = 20) -> List[str]:
    """HPLT/HPLT2.0_cleaned kor_Hang (2단계 병렬 필터링, 개수 제한 없음)"""

    print("\n" + "=" * 80)
    print(f"📊 HPLT Korean (고속 병렬 필터링, 전체 데이터)")
    print("=" * 80)

    dataset = load_dataset("HPLT/HPLT2.0_cleaned", "kor_Hang", split="train", streaming=True)

    # 1단계: 빠른 수집 (길이만 체크)
    print(f"[1/2] 빠른 수집 (길이 필터만)")

    raw_texts = []
    scanned = 0
    start_time = time.time()

    for sample in dataset:
        scanned += 1
        text = sample['text']

        # 길이 필터만 (초고속)
        if 800 <= len(text) <= 2500:
            raw_texts.append(text)

        # 진행 상황
        if scanned % 50000 == 0:
            elapsed = time.time() - start_time
            rate = scanned / elapsed
            print(f"  스캔: {scanned:,} | 수집: {len(raw_texts):,} | 속도: {rate:.0f} docs/s")

    elapsed1 = time.time() - start_time
    print(f"✅ 1단계 완료: {len(raw_texts):,}개 수집 ({elapsed1:.1f}초)")

    # 2단계: 병렬 품질 필터링
    print(f"\n[2/2] 병렬 품질 필터링 ({num_workers} 워커)")
    step2_start = time.time()

    with Pool(processes=num_workers) as pool:
        results = pool.map(quality_check_worker, raw_texts, chunksize=500)

    # None 제거 (개수 제한 없음)
    filtered_texts = [r for r in results if r is not None]

    elapsed2 = time.time() - step2_start
    total_elapsed = time.time() - start_time

    print(f"✅ 2단계 완료: {len(filtered_texts):,}개 선별 ({elapsed2:.1f}초)")
    print(f"   품질 통과율: {len(filtered_texts)/len(raw_texts)*100:.1f}%")
    print(f"⏱️  총 시간: {total_elapsed:.1f}초")

    return filtered_texts


def process_air_bench_news(num_workers: int = 20) -> List[str]:
    """AIR-Bench/qa_news_ko (2단계 병렬 필터링, 개수 제한 없음)"""

    print("\n" + "=" * 80)
    print(f"📊 AIR-Bench QA News (고속 병렬 필터링, 전체 데이터)")
    print("=" * 80)

    dataset = load_dataset("AIR-Bench/qa_news_ko", split="corpus_default", streaming=True)

    # 1단계: 빠른 수집 (길이만 체크)
    print(f"[1/2] 빠른 수집 (길이 필터만)")

    raw_texts = []
    scanned = 0
    start_time = time.time()

    for sample in dataset:
        scanned += 1
        text = sample['text']

        if 800 <= len(text) <= 2500:
            raw_texts.append(text)

        if scanned % 50000 == 0:
            elapsed = time.time() - start_time
            rate = scanned / elapsed
            print(f"  스캔: {scanned:,} | 수집: {len(raw_texts):,} | 속도: {rate:.0f} docs/s")

    elapsed1 = time.time() - start_time
    print(f"✅ 1단계 완료: {len(raw_texts):,}개 수집 ({elapsed1:.1f}초)")

    # 2단계: 병렬 품질 필터링
    print(f"\n[2/2] 병렬 품질 필터링 ({num_workers} 워커)")
    step2_start = time.time()

    with Pool(processes=num_workers) as pool:
        results = pool.map(quality_check_worker, raw_texts, chunksize=500)

    # None 제거 (개수 제한 없음)
    filtered_texts = [r for r in results if r is not None]

    elapsed2 = time.time() - step2_start
    total_elapsed = time.time() - start_time

    print(f"✅ 2단계 완료: {len(filtered_texts):,}개 선별 ({elapsed2:.1f}초)")
    print(f"   품질 통과율: {len(filtered_texts)/len(raw_texts)*100:.1f}%")
    print(f"⏱️  총 시간: {total_elapsed:.1f}초")

    return filtered_texts


def process_finance_news(num_workers: int = 20) -> List[str]:
    """nmixx-fin/twice_kr_finance_news_summ (병렬 품질 필터)"""

    print("\n" + "=" * 80)
    print(f"📊 Finance News (금융)")
    print("=" * 80)

    dataset = load_dataset("nmixx-fin/twice_kr_finance_news_summ", split="train")
    print(f"✅ 원본 샘플 수: {len(dataset):,}")

    # 1단계: 길이 필터 (800-2500자)
    texts = [s['text'] for s in dataset if 800 <= len(s['text']) <= 2500]
    print(f"   길이 필터 후: {len(texts):,}개")

    # 2단계: 병렬 품질 필터
    print(f"   품질 필터링 중 ({num_workers} 워커)...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(quality_check_worker, texts, chunksize=100)

    filtered = [r for r in results if r is not None]

    print(f"✅ 수집 완료: {len(filtered):,}개")
    return filtered


def process_bqa_news(num_workers: int = 20) -> List[str]:
    """nmixx-fin/twice_kr_news_bqa_cls (병렬 품질 필터)"""

    print("\n" + "=" * 80)
    print(f"📊 BQA News (경제)")
    print("=" * 80)

    dataset = load_dataset("nmixx-fin/twice_kr_news_bqa_cls", split="train")
    print(f"✅ 원본 샘플 수: {len(dataset):,}")

    # 1단계: 길이 필터 (800-2500자)
    texts = [s['text'] for s in dataset if 800 <= len(s['text']) <= 2500]
    print(f"   길이 필터 후: {len(texts):,}개")

    # 2단계: 병렬 품질 필터
    print(f"   품질 필터링 중 ({num_workers} 워커)...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(quality_check_worker, texts, chunksize=100)

    filtered = [r for r in results if r is not None]

    print(f"✅ 수집 완료: {len(filtered):,}개")
    return filtered


# ============================================================================
# 메인 처리
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entity Coref Dataset V2 생성 (개수 제한 없음)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="시퀀스 길이 (기본: 2048)")
    parser.add_argument("--output-dir", default="./prepared_datasets",
                        help="저장 디렉토리")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="워커 수")
    parser.add_argument("--model", default="kakaobank/kf-deberta-base",
                        help="토크나이저 모델")
    parser.add_argument("--max-entity-freq", type=int, default=1000,
                        help="개체당 최대 샘플 수 (기본: 1000, 편향 방지)")

    args = parser.parse_args()

    # 워커 수
    if args.num_workers is None:
        args.num_workers = max(4, cpu_count() - 4)

    print("\n" + "=" * 80)
    print("🚀 Entity Coreference Dataset V2 생성")
    print("=" * 80)
    print(f"시퀀스 길이: {args.seq_len}")
    print(f"워커 수: {args.num_workers}")
    print(f"개체당 최대 빈도: {args.max_entity_freq}")
    print(f"개수 제한: 없음 (전체 고품질 데이터 활용)")

    overall_start = time.time()

    # 1단계: 텍스트 수집 (병렬 품질 필터 적용)
    print("\n" + "=" * 80)
    print("📥 1단계: 고품질 텍스트 수집 (5개 데이터셋 + 병렬 필터)")
    print("=" * 80)

    all_texts = []

    # Dataset 1: Naver News Gen (최고 품질)
    texts1 = process_naver_news_gen(args.num_workers)
    all_texts.extend(texts1)

    # Dataset 2: HPLT Korean (대규모, 전체)
    texts2 = process_hplt_korean(args.num_workers)
    all_texts.extend(texts2)

    # Dataset 3: AIR-Bench News (100만+, 전체)
    texts3 = process_air_bench_news(args.num_workers)
    all_texts.extend(texts3)

    # Dataset 4: Finance News
    texts4 = process_finance_news(args.num_workers)
    all_texts.extend(texts4)

    # Dataset 5: BQA News
    texts5 = process_bqa_news(args.num_workers)
    all_texts.extend(texts5)

    print(f"\n✅ 총 수집된 텍스트: {len(all_texts):,}개 (고품질 필터 적용 완료)")

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

    # 3단계: Coref 샘플 생성 (개체 빈도 제한)
    print("\n" + "=" * 80)
    print("🔨 3단계: Coreference 샘플 생성 (개체 빈도 제한)")
    print("=" * 80)

    step3_start = time.time()

    entity_counter = Counter()
    all_examples = []

    for idx, text, repeated in results:
        examples = create_coref_examples_with_limit(
            text, repeated, entity_counter, args.max_entity_freq
        )
        all_examples.extend(examples)

        # 진행 상황 (10000개마다)
        if len(all_examples) % 10000 == 0:
            print(f"  생성된 샘플: {len(all_examples):,} (고유 개체: {len(entity_counter):,})")

    step3_elapsed = time.time() - step3_start

    print(f"\n✅ 생성된 샘플: {len(all_examples):,}")
    print(f"✅ 고유 개체 수: {len(entity_counter):,}")
    print(f"⏱️  생성 시간: {step3_elapsed:.1f}초")

    # 통계
    print("\n" + "=" * 80)
    print("📊 데이터셋 통계")
    print("=" * 80)

    distances = [ex['distance'] for ex in all_examples]
    targets = [ex['target'] for ex in all_examples]

    print(f"  거리 (선행사 ↔ 상호참조):")
    print(f"    평균: {np.mean(distances):.1f} 문자")
    print(f"    중앙값: {np.median(distances):.1f} 문자")
    print(f"    범위: {np.min(distances)} ~ {np.max(distances)} 문자")

    # 거리 분포
    short = sum(1 for d in distances if d < 200)
    medium = sum(1 for d in distances if 200 <= d < 800)
    long = sum(1 for d in distances if d >= 800)
    print(f"\n  거리 분포:")
    print(f"    Short (10-200자): {short:,} ({short/len(distances)*100:.1f}%)")
    print(f"    Medium (200-800자): {medium:,} ({medium/len(distances)*100:.1f}%)")
    print(f"    Long (800-2000자): {long:,} ({long/len(distances)*100:.1f}%)")

    print(f"\n  개체(target) 분포:")
    print(f"    고유 개체 수: {len(entity_counter):,}개")
    print(f"    Top 10 빈도:")
    for target, count in entity_counter.most_common(10):
        print(f"      '{target}': {count:,}개 ({count/len(targets)*100:.2f}%)")

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

    # 4단계: 토큰화 및 저장
    print("\n" + "=" * 80)
    print("💾 4단계: 토큰화 및 저장")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            max_length=args.seq_len,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=2000,
        remove_columns=["text"],
        num_proc=args.num_workers,
        desc=f"토큰화 (seq_len={args.seq_len})"
    )

    # 저장
    save_path = output_dir / f"entity_coref_v2_{args.seq_len}"
    save_path.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(save_path))

    step4_elapsed = time.time() - step4_start

    print(f"✅ 저장 완료: {save_path}")
    print(f"📊 샘플 수: {len(tokenized):,}")
    print(f"⏱️  시간: {step4_elapsed:.1f}초")

    # 메타데이터 저장
    meta_path = save_path / "metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset_type": "entity_coreference_v2",
            "version": "2.0",
            "num_samples": len(all_examples),
            "seq_len": args.seq_len,
            "created_at": datetime.now().isoformat(),
            "source_datasets": [
                "dev7halo/naver-news-summarization-ko-with-gen",
                "HPLT/HPLT2.0_cleaned (kor_Hang)",
                "AIR-Bench/qa_news_ko",
                "nmixx-fin/twice_kr_finance_news_summ",
                "nmixx-fin/twice_kr_news_bqa_cls"
            ],
            "num_unique_entities": len(entity_counter),
            "max_entity_freq": args.max_entity_freq,
            "top_entities": dict(entity_counter.most_common(20)),
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
            },
            "distance_distribution": {
                "short_10_200": short,
                "medium_200_800": medium,
                "long_800_2000": long
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"📝 메타데이터: {meta_path}")

    overall_elapsed = time.time() - overall_start

    # 최종 요약
    print("\n" + "=" * 80)
    print("✅ 데이터셋 생성 완료!")
    print("=" * 80)
    print(f"⏱️  전체 시간: {overall_elapsed:.1f}초 ({overall_elapsed/60:.1f}분)")
    print(f"📊 생성된 샘플: {len(all_examples):,}개")
    print(f"🎯 고유 개체 수: {len(entity_counter):,}개")
    print(f"💾 저장 경로: {save_path}")

    print("\n📖 다음 단계:")
    print("=" * 80)
    print("훈련 실행:")
    print()
    print(f"PYTHONNOUSERSITE=1 uv run python scripts/run_entity_coref_finetune.py \\")
    print(f"  --checkpoint runs/combined_experiment/checkpoint-1230 \\")
    print(f"  --dataset {save_path} \\")
    print(f"  --epochs 5 \\")
    print(f"  --batch-size 8 \\")
    print(f"  --gradient-accumulation 8 \\")
    print(f"  --lr 1.5e-5 \\")
    print(f"  --eval-steps 200 \\")
    print(f"  --output-dir runs/entity_coref_v2")


if __name__ == "__main__":
    main()
