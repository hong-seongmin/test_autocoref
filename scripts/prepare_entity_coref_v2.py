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

import sys
import json
import gc
import random
import time
import re
import tempfile
import pickle
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional, Iterable
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

    return find_exact_repetitions_from_tokens(tokens)


def find_exact_repetitions_from_tokens(tokens: list) -> Dict[str, List[int]]:
    """
    토큰 결과에서 반복 개체 찾기 (캐싱 최적화)

    Args:
        tokens: Kiwi tokenize() 결과 또는 token dict 리스트

    Returns:
        {"홍길동": [0, 45, 100], "삼성전자": [200, 350]}
    """
    # 대명사 체크 (강화)
    PRONOUN_POS = {"NP"}

    # Token 객체와 dict 모두 지원
    def get_tag(tk):
        return tk.tag if hasattr(tk, 'tag') else tk['tag']

    has_pronoun = any(get_tag(tk) in PRONOUN_POS for tk in tokens)
    if has_pronoun:
        return {}

    entity_positions = {}

    for token in tokens:
        # Token 객체와 dict 모두 지원
        tag = token.tag if hasattr(token, 'tag') else token['tag']
        form = token.form if hasattr(token, 'form') else token['form']
        start = token.start if hasattr(token, 'start') else token['start']

        # NNP (고유명사)만
        if tag == 'NNP' and len(form) >= 2:
            entity = form

            # 너무 일반적인 단어 제외
            if entity in ['것', '수', '등', '때', '곳', '점', '년', '월', '일']:
                continue

            if entity not in entity_positions:
                entity_positions[entity] = []
            entity_positions[entity].append(start)

    # 2번 이상 등장한 것만
    repeated = {e: pos for e, pos in entity_positions.items() if len(pos) >= 2}

    return repeated


def worker_find_repetitions(args):
    """멀티프로세싱용 워커 (Kiwi 토큰화 포함)"""
    text, idx = args

    # 프로세스별 Kiwi 인스턴스
    if not hasattr(worker_find_repetitions, '_kiwi'):
        worker_find_repetitions._kiwi = Kiwi()

    kiwi = worker_find_repetitions._kiwi

    repeated = find_exact_repetitions(text, kiwi)

    if repeated:
        return (idx, text, repeated)
    return None


def worker_find_repetitions_from_tokens(args):
    """
    멀티프로세싱용 워커 (캐시된 토큰 사용, Kiwi 호출 없음)

    Args:
        (text, tokens, idx)
    """
    text, tokens, idx = args

    repeated = find_exact_repetitions_from_tokens(tokens)

    if repeated:
        return (idx, text, repeated)
    return None


# ============================================================================
# 병렬 품질 필터링
# ============================================================================

# 빠른 사전 필터 (Kiwi 호출 전 30-40% 걸러냄)
_HANGUL_PATTERN = re.compile('[가-힣]')
_SENTENCE_DELIM = '.?!'

def fast_prefilter(text: str) -> bool:
    """
    초고속 사전 필터 (정규식 기반, Kiwi보다 100배 빠름)

    Returns:
        True: Kiwi 토큰화 필요, False: 즉시 탈락
    """
    # 1. 문장 수 체크 (한 번에)
    sentence_count = sum(text.count(c) for c in _SENTENCE_DELIM)
    if sentence_count < 3:
        return False

    # 2. 한글 비율 체크 (정규식)
    hangul_chars = len(_HANGUL_PATTERN.findall(text))
    hangul_ratio = hangul_chars / len(text) if len(text) > 0 else 0

    # 뉴스 텍스트는 한글 비율 40% 이상
    if hangul_ratio < 0.4:
        return False

    # 3. 숫자/특수문자 과다 체크
    non_alnum = sum(1 for c in text if not c.isalnum() and c not in ' \n\t')
    if non_alnum / len(text) > 0.3:  # 특수문자 30% 초과
        return False

    return True  # Kiwi 토큰화 필요


# Pool initializer로 Kiwi 미리 로드 (속도 최적화)
_kiwi_instance = None

def init_kiwi_worker():
    """워커 초기화 시 Kiwi 인스턴스 생성"""
    global _kiwi_instance
    _kiwi_instance = Kiwi()


def chunk_iterable(data_list: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    """고정 크기 덩어리로 리스트를 순회"""
    for start in range(0, len(data_list), chunk_size):
        yield data_list[start:start + chunk_size]

def quality_check_worker(text: str) -> Optional[str]:
    """병렬 처리용 품질 체크 워커 (2단계 최적화)"""
    global _kiwi_instance

    try:
        # 1단계: 초고속 사전 필터 (30-40% 걸러냄)
        if not fast_prefilter(text):
            return None

        # 2단계: Kiwi NNP 비율 체크 (통과한 것만)
        tokens = _kiwi_instance.tokenize(text)
        nnp_count = sum(1 for t in tokens if t.tag == 'NNP')
        nnp_ratio = nnp_count / max(1, len(tokens))

        if nnp_ratio < 0.05 or nnp_ratio > 0.20:
            return None

        return text
    except:
        return None


# 토큰 결과 캐싱용 워커 (Stage 1 → Stage 2 재사용)
def quality_check_worker_with_tokens(text: str) -> Optional[Tuple[str, list]]:
    """
    품질 체크 + 토큰 결과 반환 (Stage 2 재사용용)

    Returns:
        (text, token_data) or None
        token_data: list of dicts (picklable)
    """
    global _kiwi_instance

    try:
        # 1단계: 초고속 사전 필터
        if not fast_prefilter(text):
            return None

        # 2단계: Kiwi 토큰화
        tokens = _kiwi_instance.tokenize(text)
        nnp_count = sum(1 for t in tokens if t.tag == 'NNP')
        nnp_ratio = nnp_count / max(1, len(tokens))

        if nnp_ratio < 0.05 or nnp_ratio > 0.20:
            return None

        # Token 객체를 picklable dict로 변환 (multiprocessing 지원)
        token_data = [
            {
                'form': t.form,
                'tag': t.tag,
                'start': t.start,
                'len': t.len
            }
            for t in tokens
        ]

        return (text, token_data)
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

    # 2단계: 병렬 품질 필터 (initializer 추가로 버그 수정)
    print(f"   품질 필터링 중 ({num_workers} 워커)...")
    with Pool(processes=num_workers, initializer=init_kiwi_worker, maxtasksperchild=1000) as pool:
        results = pool.map(quality_check_worker, texts, chunksize=100)

    filtered = [r for r in results if r is not None]

    print(f"✅ 수집 완료: {len(filtered):,}개")
    return filtered


def process_hplt_korean(num_workers: int = 20) -> List[str]:
    """
    HPLT/HPLT2.0_cleaned kor_Hang (스트리밍 + 디스크 저장, OOM 완전 방지)

    Returns:
        List[str] - temp file paths (토큰 캐싱 데이터)
    """

    print("\n" + "=" * 80)
    print(f"📊 HPLT Korean (스트리밍 배치 처리, 전체 데이터)")
    print("=" * 80)

    dataset = load_dataset("HPLT/HPLT2.0_cleaned", "kor_Hang", split="train", streaming=True)

    # 스트리밍 방식: Stage 1 + 2 통합 (메모리 효율적)
    optimal_workers = min(60, num_workers)  # OOM 방지: 최대 60개
    print(f"  워커: {optimal_workers}개, 배치당 100k개 (메모리 안전)")
    print(f"  💾 배치 결과를 임시 파일로 저장 (메모리 절약)")

    BATCH_SIZE = 100_000  # 10만개씩 배치 처리
    temp_files = []  # 임시 파일 경로 저장

    current_batch = []
    scanned = 0
    total_passed = 0
    batch_num = 0
    start_time = time.time()

    # Pool 1번만 생성 (전체 스트리밍 동안 재사용)
    with Pool(processes=optimal_workers, initializer=init_kiwi_worker, maxtasksperchild=1000) as pool:

        for sample in dataset:
            scanned += 1
            text = sample['text']

            # 길이 필터 (즉시)
            if 800 <= len(text) <= 2500:
                current_batch.append(text)

            # 배치가 차면 즉시 품질 필터링
            if len(current_batch) >= BATCH_SIZE:
                batch_num += 1
                batch_start_time = time.time()

                print(f"\n배치 {batch_num} | 스캔: {scanned:,} | 배치 크기: {len(current_batch):,}", flush=True)
                print(f"  품질 필터링 시작... (워커: {optimal_workers}개)", flush=True)

                # 병렬 품질 필터링 + 토큰 캐싱
                batch_filtered = []
                for idx, result in enumerate(pool.imap_unordered(quality_check_worker_with_tokens, current_batch, chunksize=500), 1):
                    if result is not None:
                        batch_filtered.append(result)

                    # 진행도 (1,000개마다 표시 - 더 자주)
                    if idx % 1_000 == 0 or idx == 1:
                        batch_elapsed = time.time() - batch_start_time
                        rate = idx / batch_elapsed if batch_elapsed > 0 else 0
                        print(f"  처리: {idx:,} / {len(current_batch):,} ({idx/len(current_batch)*100:.1f}%) | 통과: {len(batch_filtered):,} | {rate:.0f} docs/s", flush=True)

                total_passed += len(batch_filtered)

                batch_elapsed = time.time() - batch_start_time
                total_elapsed = time.time() - start_time
                avg_rate = scanned / total_elapsed if total_elapsed > 0 else 0
                pass_rate = len(batch_filtered) / len(current_batch) * 100 if current_batch else 0

                # 💾 디스크에 저장 (메모리 해제)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', prefix='hplt_batch_')
                with open(temp_file.name, 'wb') as f:
                    pickle.dump(batch_filtered, f)
                temp_files.append(temp_file.name)

                print(f"   → 배치 완료: {len(batch_filtered):,}개 통과 ({pass_rate:.1f}%) | 누적: {total_passed:,}개 | 평균: {avg_rate:.0f} docs/s")
                print(f"   💾 저장: {temp_file.name}")

                # 메모리 해제 (핵심!)
                current_batch = []
                del batch_filtered
                gc.collect()

        # 마지막 배치 처리
        if current_batch:
            batch_num += 1
            print(f"\n마지막 배치 {batch_num} | 배치 크기: {len(current_batch):,}", flush=True)

            batch_filtered = []
            for result in pool.imap_unordered(quality_check_worker_with_tokens, current_batch, chunksize=2000):
                if result is not None:
                    batch_filtered.append(result)

            total_passed += len(batch_filtered)

            pass_rate = len(batch_filtered) / len(current_batch) * 100 if current_batch else 0

            # 💾 디스크에 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', prefix='hplt_batch_')
            with open(temp_file.name, 'wb') as f:
                pickle.dump(batch_filtered, f)
            temp_files.append(temp_file.name)

            print(f"   → 마지막 배치 완료: {len(batch_filtered):,}개 통과 ({pass_rate:.1f}%)")
            print(f"   💾 저장: {temp_file.name}")

            del current_batch, batch_filtered
            gc.collect()

    total_elapsed = time.time() - start_time
    overall_pass_rate = total_passed / scanned * 100 if scanned > 0 else 0

    print(f"\n✅ HPLT 완료: {total_passed:,}개 선별 (임시 파일 {len(temp_files)}개)")
    print(f"   스캔: {scanned:,}개 | 통과율: {overall_pass_rate:.2f}%")
    print(f"⏱️  총 시간: {total_elapsed:.1f}초 ({total_elapsed/60:.1f}분)")

    return temp_files


def process_air_bench_news(num_workers: int = 20) -> List[str]:
    """
    AIR-Bench/qa_news_ko (스트리밍 + 디스크 저장, OOM 완전 방지)

    Returns:
        List[str] - temp file paths (토큰 캐싱 데이터)
    """

    print("\n" + "=" * 80)
    print(f"📊 AIR-Bench QA News (스트리밍 배치 처리, 전체 데이터)")
    print("=" * 80)

    dataset = load_dataset("AIR-Bench/qa_news_ko", split="corpus_default", streaming=True)

    # 스트리밍 방식: Stage 1 + 2 통합 (메모리 효율적)
    optimal_workers = min(60, num_workers)  # OOM 방지: 최대 60개
    print(f"  워커: {optimal_workers}개, 배치당 100k개 (메모리 안전)")
    print(f"  💾 배치 결과를 임시 파일로 저장 (메모리 절약)")

    BATCH_SIZE = 100_000  # 10만개씩 배치 처리
    temp_files = []  # 임시 파일 경로 저장

    current_batch = []
    scanned = 0
    total_passed = 0
    batch_num = 0
    start_time = time.time()

    # Pool 1번만 생성 (전체 스트리밍 동안 재사용)
    with Pool(processes=optimal_workers, initializer=init_kiwi_worker, maxtasksperchild=1000) as pool:

        for sample in dataset:
            scanned += 1
            text = sample['text']

            # 길이 필터 (즉시)
            if 800 <= len(text) <= 2500:
                current_batch.append(text)

            # 배치가 차면 즉시 품질 필터링
            if len(current_batch) >= BATCH_SIZE:
                batch_num += 1
                batch_start_time = time.time()

                print(f"\n배치 {batch_num} | 스캔: {scanned:,} | 배치 크기: {len(current_batch):,}", flush=True)
                print(f"  품질 필터링 시작... (워커: {optimal_workers}개)", flush=True)

                # 병렬 품질 필터링 + 토큰 캐싱
                batch_filtered = []
                for idx, result in enumerate(pool.imap_unordered(quality_check_worker_with_tokens, current_batch, chunksize=500), 1):
                    if result is not None:
                        batch_filtered.append(result)

                    # 진행도 (1,000개마다 표시 - 더 자주)
                    if idx % 1_000 == 0 or idx == 1:
                        batch_elapsed = time.time() - batch_start_time
                        rate = idx / batch_elapsed if batch_elapsed > 0 else 0
                        print(f"  처리: {idx:,} / {len(current_batch):,} ({idx/len(current_batch)*100:.1f}%) | 통과: {len(batch_filtered):,} | {rate:.0f} docs/s", flush=True)

                total_passed += len(batch_filtered)

                batch_elapsed = time.time() - batch_start_time
                total_elapsed = time.time() - start_time
                avg_rate = scanned / total_elapsed if total_elapsed > 0 else 0
                pass_rate = len(batch_filtered) / len(current_batch) * 100 if current_batch else 0

                # 💾 디스크에 저장 (메모리 해제)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', prefix='air_batch_')
                with open(temp_file.name, 'wb') as f:
                    pickle.dump(batch_filtered, f)
                temp_files.append(temp_file.name)

                print(f"   → 배치 완료: {len(batch_filtered):,}개 통과 ({pass_rate:.1f}%) | 누적: {total_passed:,}개 | 평균: {avg_rate:.0f} docs/s")
                print(f"   💾 저장: {temp_file.name}")

                # 메모리 해제 (핵심!)
                current_batch = []
                del batch_filtered
                gc.collect()

        # 마지막 배치 처리
        if current_batch:
            batch_num += 1
            print(f"\n마지막 배치 {batch_num} | 배치 크기: {len(current_batch):,}", flush=True)

            batch_filtered = []
            for result in pool.imap_unordered(quality_check_worker_with_tokens, current_batch, chunksize=2000):
                if result is not None:
                    batch_filtered.append(result)

            total_passed += len(batch_filtered)

            pass_rate = len(batch_filtered) / len(current_batch) * 100 if current_batch else 0

            # 💾 디스크에 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', prefix='air_batch_')
            with open(temp_file.name, 'wb') as f:
                pickle.dump(batch_filtered, f)
            temp_files.append(temp_file.name)

            print(f"   → 마지막 배치 완료: {len(batch_filtered):,}개 통과 ({pass_rate:.1f}%)")
            print(f"   💾 저장: {temp_file.name}")

            del current_batch, batch_filtered
            gc.collect()

    total_elapsed = time.time() - start_time
    overall_pass_rate = total_passed / scanned * 100 if scanned > 0 else 0

    print(f"\n✅ AIR-Bench 완료: {total_passed:,}개 선별 (임시 파일 {len(temp_files)}개)")
    print(f"   스캔: {scanned:,}개 | 통과율: {overall_pass_rate:.2f}%")
    print(f"⏱️  총 시간: {total_elapsed:.1f}초 ({total_elapsed/60:.1f}분)")

    return temp_files


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

    # 2단계: 병렬 품질 필터 (initializer 추가로 버그 수정)
    print(f"   품질 필터링 중 ({num_workers} 워커)...")
    with Pool(processes=num_workers, initializer=init_kiwi_worker, maxtasksperchild=1000) as pool:
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

    # 2단계: 병렬 품질 필터 (initializer 추가로 버그 수정)
    print(f"   품질 필터링 중 ({num_workers} 워커)...")
    with Pool(processes=num_workers, initializer=init_kiwi_worker, maxtasksperchild=1000) as pool:
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
    parser.add_argument("--max-samples", type=int, default=None,
                        help="최대 수집 샘플 수 (기본: 제한 없음)")
    parser.add_argument("--token-chunk-size", type=int, default=20000,
                        help="토큰화 단계에서 한 번에 처리할 샘플 수 (기본: 20,000)")
    parser.add_argument("--tokenizer-batch-size", type=int, default=4096,
                        help="토크나이저 호출 배치 크기 (기본: 4,096)")

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
    if args.max_samples is not None:
        print(f"샘플 수 제한: {args.max_samples:,}개")
    else:
        print(f"개수 제한: 없음 (전체 고품질 데이터 활용)")

    overall_start = time.time()

    # 1단계: 텍스트 수집 (병렬 품질 필터 + 토큰 캐싱)
    print("\n" + "=" * 80)
    print("📥 1단계: 고품질 텍스트 수집 (5개 데이터셋 + 병렬 필터 + 토큰 캐싱)")
    print("=" * 80)

    temp_files = []  # 임시 파일 경로 (HPLT + AIR-Bench)
    plain_texts = []  # text만 저장 (Naver, Finance, BQA)

    # Dataset 1: Naver News Gen (최고 품질, text만)
    texts1 = process_naver_news_gen(args.num_workers)
    plain_texts.extend(texts1)

    # Dataset 2: HPLT Korean (대규모, 전체, 디스크 저장)
    temp_files_hplt = process_hplt_korean(args.num_workers)
    temp_files.extend(temp_files_hplt)

    # Dataset 3: AIR-Bench News (100만+, 전체, 디스크 저장)
    temp_files_air = process_air_bench_news(args.num_workers)
    temp_files.extend(temp_files_air)

    # Dataset 4: Finance News (text만)
    texts4 = process_finance_news(args.num_workers)
    plain_texts.extend(texts4)

    # Dataset 5: BQA News (text만)
    texts5 = process_bqa_news(args.num_workers)
    plain_texts.extend(texts5)

    print(f"\n✅ 총 수집된 텍스트:")
    print(f"   토큰 캐싱 (임시 파일): {len(temp_files)}개 파일 (HPLT + AIR-Bench)")
    print(f"   일반 텍스트: {len(plain_texts):,}개 (Naver + Finance + BQA)")

    # 2단계: 반복 개체 찾기 (토큰 캐싱 최적화)
    print("\n" + "=" * 80)
    print(f"🔍 2단계: 반복 개체 찾기 (토큰 캐싱 활용, Kiwi 재호출 제거)")
    print("=" * 80)

    step2_start = time.time()

    results = []

    # 2-1: 토큰 캐시된 데이터 처리 (Kiwi 호출 없음, 초고속)
    print(f"  [2-1] 토큰 캐시 활용: {len(temp_files)}개 파일 처리 (Kiwi 재호출 없음)")
    cache_start = time.time()

    idx_counter = 0
    for file_idx, temp_file in enumerate(temp_files, 1):
        print(f"    파일 {file_idx}/{len(temp_files)}: {temp_file}")

        # 파일에서 배치 로드
        with open(temp_file, 'rb') as f:
            batch_data = pickle.load(f)

        # 병렬 처리
        with Pool(processes=args.num_workers, maxtasksperchild=1000) as pool:
            batch_results = pool.map(
                worker_find_repetitions_from_tokens,
                [(text, tokens, idx_counter + i) for i, (text, tokens) in enumerate(batch_data)],
                chunksize=500
            )

        batch_results = [r for r in batch_results if r is not None]
        results.extend(batch_results)

        print(f"      → {len(batch_results):,}개 발견")

        idx_counter += len(batch_data)

        # 메모리 해제
        del batch_data, batch_results
        gc.collect()

        # 임시 파일 삭제
        os.unlink(temp_file)

    cache_elapsed = time.time() - cache_start
    print(f"     총 {len(results):,}개 발견 ({cache_elapsed:.1f}초)")

    # 2-2: 일반 텍스트 처리 (Kiwi 토큰화 필요)
    print(f"  [2-2] 일반 텍스트: {len(plain_texts):,}개 처리 (Kiwi 토큰화)")
    plain_start = time.time()

    with Pool(processes=args.num_workers, initializer=init_kiwi_worker, maxtasksperchild=1000) as pool:
        plain_results = pool.map(
            worker_find_repetitions,
            [(text, idx_counter + i) for i, text in enumerate(plain_texts)],
            chunksize=500
        )

    plain_results = [r for r in plain_results if r is not None]
    plain_elapsed = time.time() - plain_start
    print(f"     → {len(plain_results):,}개 발견 ({plain_elapsed:.1f}초)")

    results.extend(plain_results)

    step2_elapsed = time.time() - step2_start

    total_texts = idx_counter + len(plain_texts)
    print(f"\n✅ 반복 개체 발견: {len(results):,}/{total_texts:,} 텍스트 "
          f"({len(results)/total_texts*100:.1f}%)")
    print(f"⏱️  총 시간: {step2_elapsed:.1f}초 (토큰 캐싱으로 {idx_counter:,}개 Kiwi 재호출 제거)")

    # 3단계: Coref 샘플 생성 (개체 빈도 제한)
    print("\n" + "=" * 80)
    print("🔨 3단계: Coreference 샘플 생성 (개체 빈도 제한)")
    print("=" * 80)

    step3_start = time.time()

    entity_counter = Counter()
    all_examples = []

    for idx, text, repeated in results:
        # max_samples 제한 확인
        if args.max_samples is not None and len(all_examples) >= args.max_samples:
            print(f"  ⚠️  최대 샘플 수 도달: {args.max_samples:,}개 - 수집 중단", flush=True)
            break

        examples = create_coref_examples_with_limit(
            text, repeated, entity_counter, args.max_entity_freq
        )
        all_examples.extend(examples)

        # 진행 상황 (10000개마다)
        if len(all_examples) % 10000 == 0:
            print(f"  생성된 샘플: {len(all_examples):,} (고유 개체: {len(entity_counter):,})", flush=True)

            # max_samples 근접 시 경고
            if args.max_samples is not None and len(all_examples) >= args.max_samples * 0.9:
                remaining = args.max_samples - len(all_examples)
                print(f"  ℹ️  목표까지 남은 샘플: {remaining:,}개", flush=True)

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

    # 4단계: 토큰화 및 저장 (배치 처리로 OOM 방지)
    print("\n" + "=" * 80)
    print("💾 4단계: 토큰화 및 저장 (배치 처리)")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step4_start = time.time()

    total_samples = len(all_examples)

    print(f"총 샘플 수: {total_samples:,}개")

    # 배치 크기 설정 (메모리/속도 균형)
    token_chunk_size = max(1, args.token_chunk_size)
    tokenizer_batch_size = max(1, args.tokenizer_batch_size)
    num_tok_batches = (total_samples + token_chunk_size - 1) // token_chunk_size

    print(
        f"배치 수: {num_tok_batches}개 "
        f"(chunk={token_chunk_size:,}, tokenizer_batch={tokenizer_batch_size:,})"
    )

    # 임시 디렉토리 (배치 저장)
    temp_dir = output_dir / f"temp_entity_coref_v2_{args.seq_len}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    tokenized_batches = []

    # 토크나이저 미리 설정 (fast tokenizer 내부 멀티스레딩 활용)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    default_token_type = [0] * args.seq_len

    for batch_idx, batch_examples in enumerate(
        chunk_iterable(all_examples, token_chunk_size)
    ):
        current_count = len(batch_examples)
        print(
            f"\n배치 {batch_idx + 1}/{num_tok_batches}: {current_count:,}개 토큰화 중...",
            flush=True,
        )

        chunk_input_ids = []
        chunk_attention_mask = []
        chunk_token_type_ids = []

        for sub_start in range(0, current_count, tokenizer_batch_size):
            sub_examples = batch_examples[sub_start:sub_start + tokenizer_batch_size]
            sub_texts = [ex["text"] for ex in sub_examples]

            encoded = tokenizer(
                sub_texts,
                truncation=True,
                padding="max_length",
                max_length=args.seq_len,
            )

            chunk_input_ids.extend(encoded["input_ids"])
            chunk_attention_mask.extend(encoded["attention_mask"])

            if "token_type_ids" in encoded:
                chunk_token_type_ids.extend(encoded["token_type_ids"])
            else:
                chunk_token_type_ids.extend(
                    [default_token_type[:] for _ in range(len(sub_examples))]
                )

        chunk_targets = [ex["target"] for ex in batch_examples]
        chunk_antecedent = [ex["antecedent_pos"] for ex in batch_examples]
        chunk_coref = [ex["coref_pos"] for ex in batch_examples]
        chunk_distance = [ex["distance"] for ex in batch_examples]

        batch_dataset = Dataset.from_dict(
            {
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attention_mask,
                "token_type_ids": chunk_token_type_ids,
                "target": chunk_targets,
                "antecedent_pos": chunk_antecedent,
                "coref_pos": chunk_coref,
                "distance": chunk_distance,
            }
        )

        batch_path = temp_dir / f"batch_{batch_idx}"
        batch_dataset.save_to_disk(str(batch_path))
        tokenized_batches.append(str(batch_path))

        print(f"  → 배치 {batch_idx + 1} 완료: {len(batch_dataset):,}개", flush=True)

        # 메모리 정리
        del (
            batch_dataset,
            chunk_input_ids,
            chunk_attention_mask,
            chunk_token_type_ids,
            chunk_targets,
            chunk_antecedent,
            chunk_coref,
            chunk_distance,
        )
        gc.collect()

    # 배치 병합
    print(f"\n배치 병합 중...", flush=True)
    from datasets import concatenate_datasets, load_from_disk

    merged_datasets = [load_from_disk(batch_path) for batch_path in tokenized_batches]
    tokenized = concatenate_datasets(merged_datasets)

    # 최종 저장
    save_path = output_dir / f"entity_coref_v2_{args.seq_len}"
    save_path.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(save_path))

    # 임시 파일 정리
    import shutil
    shutil.rmtree(temp_dir)

    step4_elapsed = time.time() - step4_start

    print(f"\n✅ 저장 완료: {save_path}")
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
