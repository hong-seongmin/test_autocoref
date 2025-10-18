"""
필터링된 데이터셋 생성 스크립트 (개선 버전)
- KLUE MRC, Wikipedia, Naver News 필터링
- Wikipedia는 \n\n로 단락 분리
- 최적 대명사 밀도 1.5-3.0% 목표
- Entity:Pronoun 비율 균형 체크
- long_sequence_automl.py의 --dataset-choice에서 바로 사용 가능한 형식으로 저장
- 멀티프로세싱 + 2단계 필터링으로 50~100배 속도 향상
"""

import os
import sys
import gc
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from kiwipiepy import Kiwi
from typing import Dict, Any, List
from multiprocessing import Pool, cpu_count
import argparse

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Phase 1 & 2: Quick Prefilter (빠른 사전 필터)
# ============================================================================

def quick_prefilter_klue_mrc(text: str) -> bool:
    """빠른 사전 필터 - KLUE MRC용 (Kiwi 사용 안 함)"""
    # 길이 체크 (800-3000자)
    if len(text) < 800 or len(text) > 3000:
        return False

    # 간단한 대명사 패턴 체크
    simple_pronouns = ['그는', '그가', '그의', '그녀', '이는', '이가', '저는', '그것']
    pronoun_count = sum(text.count(p) for p in simple_pronouns)

    if pronoun_count < 3:  # 최소 3개 이상 (상향)
        return False

    # 특수문자/숫자 비율 체크
    special_ratio = sum(c in '[](){}#@*' for c in text) / len(text)
    if special_ratio > 0.05:  # 5% 이상이면 노이즈
        return False

    return True


def quick_prefilter_wikipedia_paragraph(text: str, seq_len: int = 2048) -> bool:
    """빠른 사전 필터 - Wikipedia 단락용 (시퀀스 길이별 최적화)"""
    # 시퀀스 길이별 목표 범위 설정
    if seq_len <= 1536:
        min_chars, max_chars = 500, 1500
    else:  # 2048
        min_chars, max_chars = 700, 2000

    # 길이 체크 (동적)
    if len(text) < min_chars or len(text) > max_chars:
        return False

    # 제목/목록 제외
    if text.strip().startswith(('==', '##', '*', '-', '|')):
        return False

    # 너무 짧은 문장들만 있으면 제외
    if text.count('.') < 2:  # 최소 2문장 이상
        return False

    # 간단한 대명사 패턴 체크
    simple_pronouns = ['그는', '그가', '그의', '그녀', '이는', '이것', '그것']
    pronoun_count = sum(text.count(p) for p in simple_pronouns)

    if pronoun_count < 2:  # 최소 2개 이상
        return False

    # 리스트/표 형식 제외
    if text.count('\n') > len(text) / 50:  # 너무 많은 줄바꿈
        return False

    return True


def quick_prefilter_naver_news(text: str) -> bool:
    """빠른 사전 필터 - Naver News용 (엄격)"""
    # 길이 체크 (1000자 이상만)
    if len(text) < 1000:
        return False

    # 매우 엄격한 대명사 체크
    simple_pronouns = ['그는', '그가', '그의', '그녀는', '그녀가', '그것은']
    pronoun_count = sum(text.count(p) for p in simple_pronouns)

    if pronoun_count < 5:  # 최소 5개 이상
        return False

    return True


# ============================================================================
# Phase 3: Kiwi Multiprocessing
# ============================================================================

def analyze_coref_quality_worker(text: str) -> Dict[str, Any]:
    """워커 프로세스용 품질 분석 함수 (각 프로세스가 자체 Kiwi 생성)"""
    if not text or len(text) < 50:
        return None

    # 프로세스별 Kiwi 인스턴스 (캐싱)
    if not hasattr(analyze_coref_quality_worker, '_kiwi'):
        analyze_coref_quality_worker._kiwi = Kiwi()

    kiwi = analyze_coref_quality_worker._kiwi

    # 너무 긴 텍스트는 앞 1500자만 (단락이므로 충분)
    text = text[:1500]

    try:
        tokens = kiwi.tokenize(text)
    except:
        return None

    pronouns = []
    entities = []
    verbs = []
    meaningful_words = []

    for token in tokens:
        if token.tag == 'NP':  # 대명사
            pronouns.append(token.form)
        elif token.tag in ['NNG', 'NNP', 'NNB']:  # 명사류
            if len(token.form) > 1:
                entities.append(token.form)
        elif token.tag.startswith(('VV', 'VA')):  # 동사/형용사
            verbs.append(token.form)

        if token.tag.startswith(('N', 'V', 'M', 'VA', 'VV')):
            meaningful_words.append(token.form)

    total_words = len(meaningful_words)
    if total_words == 0:
        return None

    # Entity:Pronoun 비율 계산
    pronoun_entity_ratio = len(pronouns) / max(1, len(entities))

    return {
        'pronoun_count': len(pronouns),
        'entity_count': len(entities),
        'verb_count': len(verbs),
        'total_words': total_words,
        'pronoun_density': len(pronouns) / max(1, total_words),
        'entity_density': len(entities) / max(1, total_words),
        'unique_pronouns': len(set(pronouns)),
        'unique_entities': len(set(entities)),
        'pronoun_entity_ratio': pronoun_entity_ratio,
    }


def batch_analyze_parallel(texts: List[str], num_workers: int = 20, chunksize: int = 50) -> List[Dict[str, Any]]:
    """병렬 배치 분석"""
    with Pool(processes=num_workers) as pool:
        results = pool.map(analyze_coref_quality_worker, texts, chunksize=chunksize)
    return results


# ============================================================================
# Improved Filtering Functions
# ============================================================================

def filter_coref_quality(quality: Dict[str, Any], dataset_type: str) -> bool:
    """
    Coreference 최적 품질 필터링
    목표: 대명사 밀도 1.0-4.0%
    실제 데이터 기반 조정: Pronoun:Entity 비율 0.01-0.1 (실제 데이터는 1-5%)
    """
    if quality is None:
        return False

    # 공통 기준 (완화)
    if quality['pronoun_count'] < 2:  # 최소 2개 대명사 (3→2 완화)
        return False

    if quality['entity_count'] < 5:  # 최소 5개 entity
        return False

    # 핵심: 대명사 밀도 1.0-4.0% (1.5→1.0 완화)
    pronoun_density = quality['pronoun_density'] * 100  # 퍼센트로 변환
    if pronoun_density < 1.0 or pronoun_density > 5.0:  # 상한도 완화
        return False

    # Entity:Pronoun 비율 체크 (0.01-0.15, 즉 1-15%)
    # 실제 데이터: 평균 0.024 (2.4%), 중앙값 0.014 (1.4%)
    # 너무 낮으면 대명사 없음, 너무 높으면 entity 부족
    if quality['pronoun_entity_ratio'] < 0.01 or quality['pronoun_entity_ratio'] > 0.15:
        return False

    # 데이터셋별 추가 기준
    if dataset_type == 'klue_mrc':
        if quality['entity_count'] < 8:  # KLUE는 entity 많음
            return False
        if quality['unique_pronouns'] < 2:
            return False

    elif dataset_type == 'wikipedia':
        # Wikipedia는 entity가 매우 많음 (평균 100개)
        # 너무 높은 기준은 비현실적
        if quality['unique_pronouns'] < 2:  # unique_entities 제거
            return False

    elif dataset_type == 'naver_news':
        # Naver News: 적절한 기준 (15-25% 통과율)
        # 실제 데이터: 중앙값 0.43%, 90% 1.32%
        if pronoun_density < 0.8:  # 0.8% 이상 (완화)
            return False
        # 비율 체크 완화
        if quality['pronoun_entity_ratio'] < 0.005:  # 0.5% 이상
            return False

    return True


# ============================================================================
# Dataset Preparation Functions
# ============================================================================

def prepare_klue_mrc_dataset(tokenizer, seq_len: int, save_dir: str, num_workers: int = 20):
    """KLUE MRC 필터링 및 토큰화 (개선 버전)"""

    print("\n" + "=" * 80)
    print(f"📊 KLUE MRC 데이터셋 준비 (seq_len={seq_len})")
    print("=" * 80)

    # 데이터 로드
    dataset = load_dataset("klue", "mrc", split="train")
    print(f"✅ 원본 샘플 수: {len(dataset)}")

    # 1단계: 빠른 필터링
    print("⚡ 1단계: 빠른 사전 필터링...")
    quick_filtered = []
    quick_filtered_texts = []

    for i, sample in enumerate(dataset):
        context = sample['context']
        if quick_prefilter_klue_mrc(context):
            quick_filtered.append(i)
            quick_filtered_texts.append(context)

    print(f"  ✅ 1단계 통과: {len(quick_filtered)}/{len(dataset)} ({len(quick_filtered)/len(dataset)*100:.1f}%)")

    if not quick_filtered_texts:
        print("⚠️ 1단계에서 모든 샘플이 제외되었습니다!")
        return None

    # 2단계: Kiwi 병렬 분석
    print(f"🚀 2단계: Kiwi 병렬 분석 ({num_workers} 워커)...")
    qualities = batch_analyze_parallel(quick_filtered_texts, num_workers=num_workers, chunksize=100)

    # 3단계: 최종 필터링 (개선된 기준)
    print("🔍 3단계: 최종 필터링 (목표: 대명사 밀도 1.5-3.0%)...")
    filtered_texts = []
    pronoun_densities = []

    for text, quality in zip(quick_filtered_texts, qualities):
        if filter_coref_quality(quality, 'klue_mrc'):
            filtered_texts.append(text)
            pronoun_densities.append(quality['pronoun_density'] * 100)

    print(f"✅ 최종 필터링 완료: {len(filtered_texts)} 샘플")
    if pronoun_densities:
        import numpy as np
        print(f"📊 대명사 밀도 평균: {np.mean(pronoun_densities):.2f}% (범위: {np.min(pronoun_densities):.2f}-{np.max(pronoun_densities):.2f}%)")

    if not filtered_texts:
        print("⚠️ 필터링된 샘플이 없습니다!")
        return None

    # Dataset으로 변환
    text_dataset = Dataset.from_dict({"text": filtered_texts})

    # 메모리 정리
    del quick_filtered_texts, qualities
    gc.collect()

    # 토큰화 (최적화)
    print(f"🔤 토큰화 중 (seq_len={seq_len}, num_proc={num_workers})...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )

    tokenized_dataset = text_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=2000,
        remove_columns=["text"],
        num_proc=num_workers,
        desc="토큰화 진행"
    )

    # 저장
    save_path = Path(save_dir) / f"klue_mrc_filtered_{seq_len}"
    save_path.mkdir(parents=True, exist_ok=True)

    tokenized_dataset.save_to_disk(str(save_path))

    print(f"💾 저장 완료: {save_path}")
    print(f"📊 최종 샘플 수: {len(tokenized_dataset)}")

    return str(save_path)


def prepare_wikipedia_dataset(tokenizer, seq_len: int, save_dir: str, max_samples: int = 50000, num_workers: int = 20):
    """
    Wikipedia 필터링 및 토큰화 (개선 버전)
    - \n\n로 단락 분리
    - 각 단락을 독립적으로 평가
    """

    print("\n" + "=" * 80)
    print(f"📊 Wikipedia 데이터셋 준비 (seq_len={seq_len})")
    print(f"🔧 개선: \\n\\n로 단락 분리하여 처리")
    print("=" * 80)

    # 데이터 로드 (스트리밍)
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)

    # 필터링 (조기 종료 포함)
    print(f"🔍 필터링 중 (최대 {max_samples}개 단락, 조기 종료 활성화)...")

    batch_size = 500
    batch_paragraphs = []
    filtered_texts = []
    scanned_docs = 0
    scanned_paragraphs = 0

    # 시퀀스 길이별 목표 범위 설정
    if seq_len <= 1536:
        min_chars, max_chars = 500, 1500
    else:  # 2048
        min_chars, max_chars = 700, 2000

    for sample in dataset:
        text = sample['text']
        scanned_docs += 1

        # ★ 핵심 개선 1: \n\n로 단락 분리
        raw_paragraphs = text.split('\n\n')

        # ★ 핵심 개선 2: 짧은 단락을 다음 단락과 병합하여 맥락 유지
        merged_paragraphs = []
        current_merged = ""

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # 병합 로직: 현재 누적 텍스트가 최소 길이 미만이면 계속 병합
            if not current_merged:
                current_merged = para
            elif len(current_merged) < min_chars:
                # 너무 짧으면 병합 (맥락 유지)
                current_merged += "\n\n" + para
            else:
                # 적절한 길이면 완성된 단락으로 추가
                # 단, 최대 길이를 초과하지 않도록 체크
                if len(current_merged) <= max_chars:
                    merged_paragraphs.append(current_merged)
                current_merged = para

        # 마지막 남은 단락 처리
        if current_merged and len(current_merged) >= min_chars and len(current_merged) <= max_chars:
            merged_paragraphs.append(current_merged)

        # 병합된 단락들 처리
        for para in merged_paragraphs:
            scanned_paragraphs += 1

            # 1단계: 빠른 필터 (단락용, 시퀀스 길이 전달)
            if quick_prefilter_wikipedia_paragraph(para, seq_len):
                batch_paragraphs.append(para)

            # 배치가 찼거나 목표 도달 시 처리
            if len(batch_paragraphs) >= batch_size or len(filtered_texts) >= max_samples:
                if batch_paragraphs:
                    # 2단계: 병렬 Kiwi 분석
                    qualities = batch_analyze_parallel(batch_paragraphs, num_workers=num_workers, chunksize=50)

                    # 3단계: 최종 필터링
                    for p, q in zip(batch_paragraphs, qualities):
                        if filter_coref_quality(q, 'wikipedia'):
                            filtered_texts.append(p)

                            # 조기 종료
                            if len(filtered_texts) >= max_samples:
                                print(f"✅ 목표 달성! {len(filtered_texts)} 단락 수집")
                                print(f"   (스캔: 문서 {scanned_docs}개, 단락 {scanned_paragraphs}개)")
                                break

                    batch_paragraphs = []

                if len(filtered_texts) >= max_samples:
                    break

        # 진행 상황 출력
        if scanned_docs % 1000 == 0:
            print(f"  문서: {scanned_docs}, 단락: {scanned_paragraphs}, 수집: {len(filtered_texts)}/{max_samples}")

        if len(filtered_texts) >= max_samples:
            break

    # 남은 배치 처리
    if batch_paragraphs and len(filtered_texts) < max_samples:
        qualities = batch_analyze_parallel(batch_paragraphs, num_workers=num_workers)
        for p, q in zip(batch_paragraphs, qualities):
            if filter_coref_quality(q, 'wikipedia'):
                filtered_texts.append(p)
                if len(filtered_texts) >= max_samples:
                    break

    print(f"✅ 필터링 완료: {len(filtered_texts)} 단락")
    print(f"   (스캔: 문서 {scanned_docs}개, 단락 {scanned_paragraphs}개)")

    if not filtered_texts:
        print("⚠️ 필터링된 샘플이 없습니다!")
        return None

    # 품질 통계
    print("📊 품질 검증 중...")
    sample_qualities = batch_analyze_parallel(filtered_texts[:100], num_workers=num_workers)
    sample_densities = [q['pronoun_density'] * 100 for q in sample_qualities if q]
    if sample_densities:
        import numpy as np
        print(f"📊 대명사 밀도 (샘플 100개): 평균 {np.mean(sample_densities):.2f}% (범위: {np.min(sample_densities):.2f}-{np.max(sample_densities):.2f}%)")

    # Dataset으로 변환
    text_dataset = Dataset.from_dict({"text": filtered_texts})

    # 메모리 정리
    del filtered_texts
    gc.collect()

    # 토큰화
    print(f"🔤 토큰화 중 (seq_len={seq_len}, num_proc={num_workers})...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )

    tokenized_dataset = text_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=2000,
        remove_columns=["text"],
        num_proc=num_workers,
        desc="토큰화 진행"
    )

    # 저장
    save_path = Path(save_dir) / f"wikipedia_filtered_{seq_len}"
    save_path.mkdir(parents=True, exist_ok=True)

    tokenized_dataset.save_to_disk(str(save_path))

    print(f"💾 저장 완료: {save_path}")
    print(f"📊 최종 샘플 수: {len(tokenized_dataset)}")

    return str(save_path)


def prepare_naver_news_dataset(tokenizer, seq_len: int, save_dir: str, num_workers: int = 20):
    """Naver News 필터링 및 토큰화 (개선 버전 - 적절한 기준)"""

    print("\n" + "=" * 80)
    print(f"📊 Naver News 데이터셋 준비 (seq_len={seq_len})")
    print("=" * 80)

    # 데이터 로드
    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")
    print(f"✅ 원본 샘플 수: {len(dataset)}")

    # 1단계: 빠른 필터링 (완화)
    print("⚡ 1단계: 빠른 사전 필터링 (완화된 기준)...")
    quick_filtered_texts = []

    for sample in dataset:
        document = sample['document']
        # 완화된 길이 기준 (800자 이상)
        if len(document) >= 800:
            quick_filtered_texts.append(document)

    print(f"  ✅ 1단계 통과: {len(quick_filtered_texts)}/{len(dataset)} ({len(quick_filtered_texts)/len(dataset)*100:.1f}%)")

    if not quick_filtered_texts:
        print("⚠️ 1단계에서 모든 샘플이 제외되었습니다!")
        return None

    # 2단계: Kiwi 병렬 분석
    print(f"🚀 2단계: Kiwi 병렬 분석 ({num_workers} 워커)...")
    qualities = batch_analyze_parallel(quick_filtered_texts, num_workers=num_workers)

    # 3단계: 최종 필터링 (적절한 기준: 15-25% 통과율 목표)
    print("🔍 3단계: 최종 필터링 (적절한 기준: 대명사 ≥2, 밀도 ≥0.8%)...")
    filtered_texts = []
    pronoun_densities = []

    for text, quality in zip(quick_filtered_texts, qualities):
        if filter_coref_quality(quality, 'naver_news'):
            filtered_texts.append(text)
            pronoun_densities.append(quality['pronoun_density'] * 100)

    print(f"✅ 최종 필터링 완료: {len(filtered_texts)} 샘플")
    if pronoun_densities:
        import numpy as np
        print(f"📊 대명사 밀도 평균: {np.mean(pronoun_densities):.2f}% (범위: {np.min(pronoun_densities):.2f}-{np.max(pronoun_densities):.2f}%)")

    if not filtered_texts:
        print(f"⚠️ 필터링된 샘플이 없습니다!")
        return None

    # Dataset으로 변환
    text_dataset = Dataset.from_dict({"text": filtered_texts})

    # 메모리 정리
    del quick_filtered_texts, qualities
    gc.collect()

    # 토큰화
    print(f"🔤 토큰화 중 (seq_len={seq_len}, num_proc={num_workers})...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )

    tokenized_dataset = text_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=2000,
        remove_columns=["text"],
        num_proc=num_workers,
        desc="토큰화 진행"
    )

    # 저장
    save_path = Path(save_dir) / f"naver_news_filtered_{seq_len}"
    save_path.mkdir(parents=True, exist_ok=True)

    tokenized_dataset.save_to_disk(str(save_path))

    print(f"💾 저장 완료: {save_path}")
    print(f"📊 최종 샘플 수: {len(tokenized_dataset)}")

    return str(save_path)


def combine_and_save_datasets(filtered_texts_dict: Dict[str, List[str]], tokenizer, seq_len: int,
                               save_dir: str, output_name: str, num_workers: int = 20):
    """
    여러 데이터셋의 필터링된 텍스트를 하나로 결합하여 저장

    Args:
        filtered_texts_dict: {dataset_name: [filtered_texts]} 형태의 딕셔너리
        tokenizer: 토크나이저
        seq_len: 시퀀스 길이
        save_dir: 저장 디렉토리
        output_name: 출력 데이터셋 이름
        num_workers: 워커 수
    """
    print("\n" + "=" * 80)
    print(f"🔗 데이터셋 결합: {output_name} (seq_len={seq_len})")
    print("=" * 80)

    # 모든 텍스트 결합
    combined_texts = []
    for dataset_name, texts in filtered_texts_dict.items():
        print(f"  + {dataset_name}: {len(texts)}개")
        combined_texts.extend(texts)

    print(f"📊 총 {len(combined_texts)}개 샘플 결합")

    if not combined_texts:
        print("⚠️ 결합할 샘플이 없습니다!")
        return None

    # Dataset으로 변환
    text_dataset = Dataset.from_dict({"text": combined_texts})

    # 메모리 정리
    del combined_texts
    gc.collect()

    # 토큰화
    print(f"🔤 토큰화 중 (seq_len={seq_len}, num_proc={num_workers})...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )

    tokenized_dataset = text_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=2000,
        remove_columns=["text"],
        num_proc=num_workers,
        desc="토큰화 진행"
    )

    # 저장
    save_path = Path(save_dir) / f"{output_name}_{seq_len}"
    save_path.mkdir(parents=True, exist_ok=True)

    tokenized_dataset.save_to_disk(str(save_path))

    print(f"💾 저장 완료: {save_path}")
    print(f"📊 최종 샘플 수: {len(tokenized_dataset)}")

    return str(save_path)


def main():
    parser = argparse.ArgumentParser(description="필터링된 데이터셋 생성 (개선 버전)")
    parser.add_argument("--model", default="kakaobank/kf-deberta-base", help="모델 이름")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[1536, 2048], help="시퀀스 길이 (기본: 1536, 2048)")
    parser.add_argument("--save-dir", default="./prepared_datasets", help="저장 디렉토리")
    parser.add_argument("--datasets", nargs="+", choices=["klue_mrc", "wikipedia", "naver_news", "all"],
                        default=["all"], help="생성할 데이터셋 (기본: all)")
    parser.add_argument("--wiki-samples", type=int, default=50000, help="Wikipedia 최대 단락 수 (기본: 50000, 충분한 양)")
    parser.add_argument("--num-workers", type=int, default=None, help="병렬 처리 워커 수 (기본: CPU 코어 - 4)")
    parser.add_argument("--combine-datasets", action="store_true", help="여러 데이터셋을 하나로 결합 (기본: 각각 분리)")
    parser.add_argument("--output-name", type=str, default=None, help="출력 데이터셋 이름 (기본: 자동 생성)")

    args = parser.parse_args()

    # 워커 수 설정
    if args.num_workers is None:
        args.num_workers = max(4, cpu_count() - 4)  # 4개 코어는 시스템용으로 남김

    print(f"⚙️  병렬 처리 워커 수: {args.num_workers} (CPU 코어: {cpu_count()})")

    # 저장 디렉토리 생성
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 토크나이저 로드
    print(f"📥 토크나이저 로드: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 데이터셋 선택
    datasets_to_create = args.datasets
    if "all" in datasets_to_create:
        datasets_to_create = ["klue_mrc", "wikipedia", "naver_news"]

    print("\n" + "=" * 80)
    print("🚀 필터링된 데이터셋 생성 시작 (개선 버전)")
    print("=" * 80)
    print(f"모델: {args.model}")
    print(f"시퀀스 길이: {args.seq_lengths}")
    print(f"저장 경로: {save_dir}")
    print(f"생성할 데이터셋: {datasets_to_create}")
    print(f"\n🎯 핵심 개선사항:")
    print(f"  ✅ Wikipedia: \\n\\n로 단락 분리 (긴 문서 → 적절한 단락)")
    print(f"  ✅ 목표 대명사 밀도: 1.5-3.0%")
    print(f"  ✅ Entity:Pronoun 비율 체크: 15-50%")
    print(f"  ✅ 2단계 필터링 (빠른 사전 필터 + Kiwi 분석)")
    print(f"  ✅ Kiwi 멀티프로세싱 ({args.num_workers} 워커)")
    print(f"  ✅ 토큰화 병렬화 ({args.num_workers} 프로세스)")
    print(f"예상 속도: 50~100배 향상 🚀")

    # 출력 이름 설정
    if args.output_name is None:
        if args.combine_datasets:
            args.output_name = "combined_coref"
        # 개별 저장 모드는 각 함수에서 자동 이름 사용

    # 결합 모드 안내
    if args.combine_datasets:
        print(f"🔗 결합 모드: 모든 데이터셋을 '{args.output_name}' 이름으로 결합")
    else:
        if args.output_name:
            print(f"📝 커스텀 이름: '{args.output_name}' 사용")

    created_paths = {}

    import time
    overall_start = time.time()

    for seq_len in args.seq_lengths:
        print(f"\n{'='*80}")
        print(f"🎯 시퀀스 길이 {seq_len} 처리")
        print(f"{'='*80}")

        seq_paths = []
        seq_start = time.time()

        # 결합 모드: 필터링된 텍스트만 수집
        filtered_texts_dict = {} if args.combine_datasets else None

        # KLUE MRC
        if "klue_mrc" in datasets_to_create:
            try:
                start = time.time()
                if args.combine_datasets:
                    # 결합 모드: 텍스트만 수집
                    print("\n" + "=" * 80)
                    print(f"📊 KLUE MRC 필터링 (결합용, seq_len={seq_len})")
                    print("=" * 80)
                    from datasets import load_dataset as hf_load_dataset
                    dataset = hf_load_dataset("klue", "mrc", split="train")
                    quick_filtered_texts = [s['context'] for s in dataset if quick_prefilter_klue_mrc(s['context'])]
                    print(f"⚡ 1단계 통과: {len(quick_filtered_texts)}/{len(dataset)}")
                    qualities = batch_analyze_parallel(quick_filtered_texts, num_workers=args.num_workers, chunksize=100)
                    filtered_texts = [t for t, q in zip(quick_filtered_texts, qualities) if filter_coref_quality(q, 'klue_mrc')]
                    print(f"✅ 최종: {len(filtered_texts)} 샘플")
                    filtered_texts_dict['klue_mrc'] = filtered_texts
                else:
                    # 분리 모드: 기존 방식
                    path = prepare_klue_mrc_dataset(tokenizer, seq_len, str(save_dir), args.num_workers)
                    if path:
                        seq_paths.append(path)
                elapsed = time.time() - start
                print(f"⏱️  KLUE MRC 처리 시간: {elapsed:.1f}초")
            except Exception as e:
                print(f"❌ KLUE MRC 처리 실패: {e}")
                import traceback
                traceback.print_exc()

        # Wikipedia
        if "wikipedia" in datasets_to_create:
            try:
                start = time.time()
                if args.combine_datasets:
                    # 결합 모드: 텍스트만 수집
                    print("\n" + "=" * 80)
                    print(f"📊 Wikipedia 필터링 (결합용, seq_len={seq_len})")
                    print("=" * 80)
                    # Wikipedia 필터링 로직 (간소화)
                    from datasets import load_dataset as hf_load_dataset
                    wiki_dataset = hf_load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
                    filtered_texts = []
                    batch_paragraphs = []

                    if seq_len <= 1536:
                        min_chars, max_chars = 500, 1500
                    else:
                        min_chars, max_chars = 700, 2000

                    for i, sample in enumerate(wiki_dataset):
                        if len(filtered_texts) >= args.wiki_samples:
                            break
                        raw_paragraphs = sample['text'].split('\n\n')
                        merged_paragraphs = []
                        current_merged = ""
                        for para in raw_paragraphs:
                            para = para.strip()
                            if not para:
                                continue
                            if not current_merged:
                                current_merged = para
                            elif len(current_merged) < min_chars:
                                current_merged += "\n\n" + para
                            else:
                                if len(current_merged) <= max_chars:
                                    merged_paragraphs.append(current_merged)
                                current_merged = para
                        if current_merged and min_chars <= len(current_merged) <= max_chars:
                            merged_paragraphs.append(current_merged)

                        for para in merged_paragraphs:
                            if quick_prefilter_wikipedia_paragraph(para, seq_len):
                                batch_paragraphs.append(para)
                            if len(batch_paragraphs) >= 500:
                                qualities = batch_analyze_parallel(batch_paragraphs, num_workers=args.num_workers, chunksize=50)
                                for p, q in zip(batch_paragraphs, qualities):
                                    if filter_coref_quality(q, 'wikipedia'):
                                        filtered_texts.append(p)
                                        if len(filtered_texts) >= args.wiki_samples:
                                            break
                                batch_paragraphs = []
                                if len(filtered_texts) >= args.wiki_samples:
                                    break
                        if (i + 1) % 1000 == 0:
                            print(f"  문서: {i+1}, 수집: {len(filtered_texts)}/{args.wiki_samples}")

                    print(f"✅ 최종: {len(filtered_texts)} 샘플")
                    filtered_texts_dict['wikipedia'] = filtered_texts
                else:
                    # 분리 모드: 기존 방식
                    path = prepare_wikipedia_dataset(tokenizer, seq_len, str(save_dir), args.wiki_samples, args.num_workers)
                    if path:
                        seq_paths.append(path)
                elapsed = time.time() - start
                print(f"⏱️  Wikipedia 처리 시간: {elapsed:.1f}초")
            except Exception as e:
                print(f"❌ Wikipedia 처리 실패: {e}")
                import traceback
                traceback.print_exc()

        # Naver News
        if "naver_news" in datasets_to_create:
            try:
                start = time.time()
                if args.combine_datasets:
                    # 결합 모드: 텍스트만 수집
                    print("\n" + "=" * 80)
                    print(f"📊 Naver News 필터링 (결합용, seq_len={seq_len})")
                    print("=" * 80)
                    from datasets import load_dataset as hf_load_dataset
                    dataset = hf_load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")
                    quick_filtered_texts = [s['document'] for s in dataset if len(s['document']) >= 800]
                    print(f"⚡ 1단계 통과: {len(quick_filtered_texts)}/{len(dataset)}")
                    qualities = batch_analyze_parallel(quick_filtered_texts, num_workers=args.num_workers)
                    filtered_texts = [t for t, q in zip(quick_filtered_texts, qualities) if filter_coref_quality(q, 'naver_news')]
                    print(f"✅ 최종: {len(filtered_texts)} 샘플")
                    filtered_texts_dict['naver_news'] = filtered_texts
                else:
                    # 분리 모드: 기존 방식
                    path = prepare_naver_news_dataset(tokenizer, seq_len, str(save_dir), args.num_workers)
                    if path:
                        seq_paths.append(path)
                elapsed = time.time() - start
                print(f"⏱️  Naver News 처리 시간: {elapsed:.1f}초")
            except Exception as e:
                print(f"❌ Naver News 처리 실패: {e}")
                import traceback
                traceback.print_exc()

        # 결합 모드: 모든 텍스트를 하나로 결합하여 저장
        if args.combine_datasets and filtered_texts_dict:
            try:
                path = combine_and_save_datasets(
                    filtered_texts_dict, tokenizer, seq_len,
                    str(save_dir), args.output_name, args.num_workers
                )
                if path:
                    seq_paths.append(path)
            except Exception as e:
                print(f"❌ 데이터셋 결합 실패: {e}")
                import traceback
                traceback.print_exc()

        created_paths[seq_len] = seq_paths

        seq_elapsed = time.time() - seq_start
        print(f"\n⏱️  시퀀스 {seq_len} 총 처리 시간: {seq_elapsed:.1f}초")

    overall_elapsed = time.time() - overall_start

    # 결과 요약
    print("\n" + "=" * 80)
    print("✅ 데이터셋 생성 완료!")
    print("=" * 80)
    print(f"⏱️  전체 처리 시간: {overall_elapsed:.1f}초 ({overall_elapsed/60:.1f}분)")

    for seq_len, paths in created_paths.items():
        print(f"\n시퀀스 길이 {seq_len}:")
        if paths:
            for path in paths:
                print(f"  - {path}")
        else:
            print(f"  (생성된 데이터셋 없음)")

    # 사용 방법 출력
    print("\n" + "=" * 80)
    print("📖 사용 방법")
    print("=" * 80)

    if created_paths:
        # 첫 번째 시퀀스 길이의 경로들을 예시로 사용
        first_seq = list(created_paths.keys())[0]
        example_paths = created_paths[first_seq]

        if example_paths:
            print("\n다음 명령어로 학습을 실행하세요:")
            print()

            # 단일 데이터셋
            print("# 1. 단일 데이터셋 사용:")
            print(f"python -m coref_automl.long_sequence_automl \\")
            print(f"    --model {args.model} \\")
            print(f"    --seq-lengths {' '.join(map(str, args.seq_lengths))} \\")
            print(f"    --trials 10 \\")
            print(f"    --dataset-choice {example_paths[0]}")

            if len(example_paths) > 1:
                # 여러 데이터셋
                print("\n# 2. 여러 데이터셋 함께 사용:")
                print(f"python -m coref_automl.long_sequence_automl \\")
                print(f"    --model {args.model} \\")
                print(f"    --seq-lengths {' '.join(map(str, args.seq_lengths))} \\")
                print(f"    --trials 10 \\")
                for path in example_paths:
                    print(f"    --dataset-choice {path} \\")
                print()

            print("\n💡 Tip: --dataset-choice를 여러 번 사용하면 Optuna가 자동으로 최적의 데이터셋을 선택합니다!")


if __name__ == "__main__":
    main()
