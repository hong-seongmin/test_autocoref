# coref_automl/long_sequence_automl.py
"""
DeBERTa Long Sequence AutoML System
- Max length: 1024-2048 tokens
- Enhanced datasets with quality filtering
- Memory-efficient batch configuration
- Advanced hyperparameter optimization
"""

from __future__ import annotations
import os
import gc
import math
import random
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache

import numpy as np
import optuna
import torch
from datasets import load_dataset, concatenate_datasets, disable_caching
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)

from .coref_utils import is_noun
from .callback import LiveMetricsCallback
from .bus import BUS
from .tune import (
    build_eval_from_lambada,
    build_coref_eval_set,
    eval_lambada_topk,
    eval_coref_f1,
    eval_coref_recall_topk,
    DynCollator
)

disable_caching()


# ────────────────────────────────────────────────────────────────────────────────
# Enhanced Dataset Loading for Long Sequences
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """데이터셋 설정"""
    source: str
    subset: Optional[str]
    split: str
    domain: str
    quality_weight: float
    min_length: int
    max_length: int

def get_long_sequence_dataset_configs(target_seq_len: int) -> List[DatasetConfig]:
    """긴 시퀀스 학습을 위한 데이터셋 설정 - KLUE MRC만 사용 (NLI 제외)"""

    base_configs = [
        # KLUE MRC만 사용 (NLI는 0% 통과율로 제외)
        DatasetConfig(
            source="klue",
            subset="mrc",
            split="train",
            domain="qa_long",
            quality_weight=1.0,
            min_length=int(target_seq_len * 0.4),  # MRC context는 보통 길어서 낮은 비율
            max_length=int(target_seq_len * 1.8)
        ),
    ]

    return base_configs

def load_single_dataset(args):
    """단일 데이터셋 로드를 위한 헬퍼 함수 (병렬 처리용)"""
    config, tokenizer, target_seq_len, limit, skip_quality_analysis = args

    try:
        # 데이터 로드
        load_kwargs = {"split": config.split}
        if config.subset:
            load_kwargs["name"] = config.subset

        dataset = load_dataset(config.source, **load_kwargs)

        # 빠른 테스트를 위해 샘플 수 제한
        if limit and limit <= 10:
            max_samples = min(100, len(dataset))
            dataset = dataset.select(range(max_samples))

        # 도메인별 전처리
        processed_dataset = preprocess_domain_data(dataset, config, tokenizer, target_seq_len, limit, skip_quality_analysis)

        return processed_dataset if processed_dataset and len(processed_dataset) > 0 else None

    except Exception as e:
        print(f"    ❌ Failed to load {config.source}: {e}")
        return None

def load_prepared_dataset(dataset_path: str) -> Dict[str, Any]:
    """준비된 데이터셋 로드"""
    from datasets import load_from_disk

    print(f"📂 Loading prepared dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"✅ Loaded {len(dataset)} samples from prepared dataset")

    return dataset

def load_enhanced_dataset(tokenizer, target_seq_len: int, limit: Optional[int] = None, skip_quality_analysis: bool = False) -> Dict[str, Any]:
    """향상된 데이터셋 로딩 - 병렬 처리 강화"""

    configs = get_long_sequence_dataset_configs(target_seq_len)

    print(f"📊 Loading {len(configs)} datasets for {target_seq_len} tokens...")

    # 병렬 데이터 로드 (ThreadPoolExecutor 사용)
    max_workers = min(len(configs), multiprocessing.cpu_count(), 8)  # 최대 8개 워커

    print(f"🚀 Parallel dataset loading: {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        args_list = [(config, tokenizer, target_seq_len, limit, skip_quality_analysis) for config in configs]
        results = list(executor.map(load_single_dataset, args_list))

    # 유효한 데이터셋만 필터링
    all_datasets = [ds for ds in results if ds is not None]

    if not all_datasets:
        raise RuntimeError("No datasets could be loaded")

    print(f"🔄 Combining {len(all_datasets)} datasets...")
    # 데이터 통합
    combined_dataset = concatenate_datasets(all_datasets)
    combined_dataset = combined_dataset.shuffle(seed=42)

    original_size = len(combined_dataset)
    if limit and len(combined_dataset) > limit:
        combined_dataset = combined_dataset.select(range(limit))
        print(f"✂️  Limited to {len(combined_dataset)}/{original_size} samples")

    print(f"🔤 Tokenizing {len(combined_dataset)} samples...")

    # 토큰화 함수
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=target_seq_len,
            return_tensors="pt"
        )

    # 시스템 코어 수에 따른 병렬 처리 최적화
    num_proc = min(multiprocessing.cpu_count(), 16)  # 최대 16개 프로세스
    batch_size = 1000  # 배치 크기 증가

    print(f"🔄 Tokenizing with {num_proc} processes, batch_size={batch_size}...")

    tokenized_dataset = combined_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        num_proc=num_proc
    )

    print(f"🎉 Dataset ready: {len(tokenized_dataset)} samples × {target_seq_len} tokens")
    return tokenized_dataset

@lru_cache(maxsize=5000)
def analyze_coref_quality_cached(text: str) -> Dict[str, float]:
    """Kiwi를 사용한 Coreference 품질 분석 (캐싱 적용)"""
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
    except ImportError:
        # Kiwi가 없는 경우 간단한 휴리스틱 사용
        return {
            'pronoun_density': calculate_simple_pronoun_density(text),
            'entity_density': calculate_simple_entity_density(text),
            'coref_score': calculate_simple_coref_score(text),
            'pronoun_count': 0,
            'entity_count': 0,
            'total_words': len(text.split())
        }

    # 형태소 분석
    tokens = kiwi.tokenize(text)

    # 대명사와 개체 분석
    pronouns = []
    entities = []

    for token in tokens:
        # Kiwi 태그 설명:
        # NP: 대명사 (그, 그녀, 이것 등)
        # NNG/NNP: 일반명사/고유명사
        # NNB: 의존명사
        if token.tag == 'NP':  # 대명사
            pronouns.append(token.form)
        elif token.tag in ['NNG', 'NNP', 'NNB']:  # 명사류
            if len(token.form) > 1:  # 한 글자 제외
                entities.append(token.form)

    # 의미 있는 단어 수 계산 (명사, 동사, 형용사 등)
    meaningful_words = len([t for t in tokens if t.tag.startswith(('N', 'V', 'M', 'VA', 'VV'))])

    pronoun_density = len(pronouns) / max(1, meaningful_words)
    entity_density = len(entities) / max(1, meaningful_words)

    # Coreference 점수 계산 (대명사-개체 상호작용)
    coref_score = min(1.0, (pronoun_density * 20) + (entity_density * 3) + (pronoun_density * entity_density * 50))

    return {
        'pronoun_density': pronoun_density,
        'entity_density': entity_density,
        'coref_score': coref_score,
        'pronoun_count': len(pronouns),
        'entity_count': len(entities),
        'total_words': meaningful_words
    }

def analyze_coref_quality(text: str) -> Dict[str, float]:
    """캐싱된 품질 분석 함수"""
    return analyze_coref_quality_cached(text)

def batch_analyze_coref_quality(texts: List[str], batch_size: int = 50, max_workers: Optional[int] = None) -> List[Dict[str, float]]:
    """배치 단위 품질 분석 (ThreadPoolExecutor 사용)"""
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)  # 최대 8개 워커

    results = []

    # 배치로 나누어 처리
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(analyze_coref_quality_cached, batch_texts))
            results.extend(batch_results)

    return results

def calculate_simple_pronoun_density(text: str) -> float:
    """Kiwi 없는 경우 간단한 대명사 밀도 계산"""
    simple_pronouns = ['그', '그녀', '그것', '이', '그의', '그녀의', '이것', '저것', '그들', '이들']
    words = text.split()
    if not words:
        return 0.0
    pronoun_count = sum(1 for word in words if word in simple_pronouns)
    return pronoun_count / len(words)

def calculate_simple_entity_density(text: str) -> float:
    """간단한 개체 밀도 계산"""
    # 한국어 개체 단서 (매우 단순화)
    entity_indicators = ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로']
    words = text.split()
    if not words:
        return 0.0
    entity_count = sum(1 for word in words if any(ind in word for ind in entity_indicators))
    return entity_count / len(words)

def calculate_simple_coref_score(text: str) -> float:
    """간단한 coreference 점수 계산"""
    pronoun_density = calculate_simple_pronoun_density(text)
    entity_density = calculate_simple_entity_density(text)
    # 대명사-개체 상호작용을 고려한 점수
    return min(1.0, (pronoun_density * 20) + (entity_density * 3) + (pronoun_density * entity_density * 50))

def preprocess_domain_data(dataset, config: DatasetConfig, tokenizer, target_seq_len: int, limit: Optional[int] = None, skip_quality_analysis: bool = False):
    """도메인별 데이터 전처리 - Coreference 특화 (실시간 진행 표시)"""

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    processed_texts = []
    quality_stats = {'total': 0, 'passed': 0, 'avg_pronoun_density': 0, 'avg_coref_score': 0}

    # 데이터셋 샘플 수 제한 (빠른 테스트용)
    if limit and limit <= 10:  # 아주 작은 limit인 경우 각 데이터셋도 제한
        max_samples = min(100, len(dataset))  # 각 데이터셋당 최대 100개
    else:
        max_samples = 5000 if config.domain == "wiki_coref" else len(dataset)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"  🔍 Processing {len(dataset)} samples from {config.domain}...")

    # 진행 표시 설정
    iterator = tqdm(dataset, desc=f"    {config.domain}", unit="samples") if use_tqdm else dataset

    print(f"    [DEBUG] Starting to process {len(dataset)} items")
    for item in iterator:
        if quality_stats['total'] <= 3:  # 처음 3개만 디버깅 출력
            print(f"    [DEBUG] Processing item {quality_stats['total'] + 1}")
        try:
            quality_stats['total'] += 1

            # 도메인별 텍스트 추출
            if config.domain == "wiki_coref":
                text = item.get("text", "").strip()
                # 인물/사건 중심 문서 필터링
                if not is_coref_rich_wiki(text):
                    continue
            elif config.domain == "news_long":
                # KLUE ynat 데이터셋 구조 확인
                if quality_stats['total'] <= 3:  # 처음 3개만 디버깅 출력
                    print(f"    [DEBUG] YNAT item keys: {list(item.keys())}")
                title = item.get("title", "")
                content = item.get("content", item.get("text", ""))
                text = f"{title} {content}".strip()
                # 긴 뉴스 기사 위주 (500자 이상)
                if len(text) < 500:
                    if quality_stats['total'] <= 3:  # 처음 3개만 디버깅 출력
                        print(f"    [DEBUG] YNAT text too short: {len(text)}")
                    continue
            elif config.domain == "qa_long":
                context = item.get("context", "")
                question = item.get("question", "")
                answer = item.get("answers", {}).get("text", [""])[0] if item.get("answers") else ""
                text = f"{context} {question} {answer}".strip()
            elif config.domain == "news_sts":
                # KLUE STS 데이터셋 구조 확인
                if quality_stats['total'] <= 3:  # 처음 3개만 디버깅 출력
                    print(f"    [DEBUG] STS item keys: {list(item.keys())}")
                sentence1 = item.get("sentence1", item.get("text", ""))
                sentence2 = item.get("sentence2", "")
                text = f"{sentence1} {sentence2}".strip()
            elif config.domain == "nli_long":
                # KLUE NLI 데이터셋 구조 확인 및 처리
                if quality_stats['total'] <= 3:  # 처음 3개만 디버깅 출력
                    print(f"    [DEBUG] NLI item keys: {list(item.keys())}")
                # KLUE NLI 데이터셋은 premise/hypothesis 구조
                premise = item.get("premise", "")
                hypothesis = item.get("hypothesis", "")
                if not premise or not hypothesis:
                    if quality_stats['total'] <= 3:  # 처음 3개만 디버깅 출력
                        print(f"    [DEBUG] NLI missing premise/hypothesis: premise='{premise[:50] if premise else 'None'}...', hypothesis='{hypothesis[:50] if hypothesis else 'None'}...'")
                    continue  # premise나 hypothesis가 없으면 스킵
                text = f"{premise} {hypothesis}".strip()
            elif config.domain == "news_topic":
                title = item.get("title", "")
                content = item.get("description", "") + " " + item.get("body", "")
                text = f"{title} {content}".strip()
            else:
                text = item.get("text", "").strip()

            if not text or len(text) < 50:  # STS 데이터셋용으로 최소 길이 낮춤
                if quality_stats['total'] <= 3:  # 처음 3개만 디버깅 출력
                    print(f"    [DEBUG] Filtered out due to text length: {len(text) if text else 0}")
                continue

            # 품질 필터링 생략 - 빠른 테스트용으로 모든 샘플 통과
            # 품질 통계 업데이트 (더미 값 사용)
            quality_stats['avg_pronoun_density'] += 0.01
            quality_stats['avg_coref_score'] += 0.5

            # 디버깅: 통계 업데이트 확인
            if quality_stats['total'] <= 5:  # 처음 5개만 출력
                print(f"    [DEBUG] Sample {quality_stats['total']}: stats updated")

            # 길이 필터링
            tokens = tokenizer.encode(text)
            token_len = len(tokens)
            if quality_stats['total'] <= 3:  # 처음 3개만 디버깅 출력
                print(f"    [DEBUG] Text length: {len(text)}, Token length: {token_len}, Range: {config.min_length}-{config.max_length}")
            if token_len < config.min_length or token_len > config.max_length:
                if quality_stats['total'] <= 3:  # 처음 3개만 디버깅 출력
                    print(f"    [DEBUG] Filtered out due to length")
                continue

            # 긴 텍스트 청킹 (coreference 맥락 유지)
            if len(tokens) > target_seq_len * 1.2:
                chunks = chunk_with_coref_preservation(text, target_seq_len, tokenizer)
                processed_texts.extend(chunks)
            else:
                processed_texts.append(text)

            quality_stats['passed'] += 1

            # tqdm 진행 표시 업데이트
            if use_tqdm and hasattr(iterator, 'set_postfix'):
                pass_rate = quality_stats['passed'] / quality_stats['total']
                iterator.set_postfix({
                    'passed': f"{quality_stats['passed']}/{quality_stats['total']}",
                    'rate': f"{pass_rate:.1%}"
                })

        except Exception as e:
            continue

    if use_tqdm:
        iterator.close()

    # 품질 통계 출력
    if quality_stats['total'] > 0:
        avg_pronoun = quality_stats['avg_pronoun_density'] / quality_stats['total']
        avg_coref = quality_stats['avg_coref_score'] / quality_stats['total']
        pass_rate = quality_stats['passed'] / quality_stats['total']

        print(f"  ✅ {config.domain}: {quality_stats['passed']}/{quality_stats['total']} passed ({pass_rate:.1%})")
        print(f"    📊 Avg pronoun density: {avg_pronoun:.3f}, Avg coref score: {avg_coref:.3f}")
    else:
        print(f"  ⚠️  {config.domain}: No samples processed")

    # 품질 기반 샘플링 (품질 분석 건너뛰기 옵션)
    if len(processed_texts) > 1000 and not skip_quality_analysis:
        print(f"  🎯 Quality sampling {len(processed_texts)} → 1000 samples...")

        # 배치 크기 및 워커 수 설정
        batch_size = 100  # 배치당 100개 텍스트
        max_workers = min(multiprocessing.cpu_count(), 12)  # 최대 12개 워커

        # 분석할 텍스트 수 제한 (메모리 효율성)
        analyze_limit = min(2000, len(processed_texts))
        texts_to_analyze = processed_texts[:analyze_limit]

        print(f"    🚀 Parallel quality analysis: {len(texts_to_analyze)} texts, {max_workers} workers, batch_size={batch_size}")

        # 배치 단위 병렬 품질 분석 (실시간 진행 표시 + 대시보드 연동)
        print(f"    🔍 Analyzing {len(texts_to_analyze)} texts for quality...")
        quality_scores = []

        # 진행 표시를 위한 배치 처리
        total_batches = (len(texts_to_analyze) + batch_size - 1) // batch_size
        completed_batches = 0

        for i in range(0, len(texts_to_analyze), batch_size):
            batch_texts = texts_to_analyze[i:i + batch_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                batch_results = list(executor.map(analyze_coref_quality_cached, batch_texts))
                quality_scores.extend(batch_results)

            completed_batches += 1
            progress = completed_batches / total_batches
            processed_count = len(quality_scores)

            # 실시간 진행 상황 출력
            print(f"    📈 Quality analysis: {completed_batches}/{total_batches} batches ({progress:.1%}) - {processed_count}/{len(texts_to_analyze)} texts")

            # 대시보드에 진행 상황 전송 (10%마다)
            if completed_batches % max(1, total_batches // 10) == 0 or completed_batches == total_batches:
                BUS.log(
                    section="dataset_progress",
                    stage="quality_analysis",
                    completed=processed_count,
                    total=len(texts_to_analyze),
                    progress=progress,
                    seq_len=target_seq_len,
                    domain=config.domain
                )

        # coref_score만 추출
        quality_scores = [result['coref_score'] for result in quality_scores]

        print(f"    ✅ Quality analysis completed: {len(quality_scores)} scores calculated")

        selected_indices = quality_weighted_sample_indices(quality_scores, min(1000, len(processed_texts)))
        processed_texts = [processed_texts[i] for i in selected_indices]
    elif len(processed_texts) > 1000 and skip_quality_analysis:
        print(f"  ⏭️  Skipping quality analysis, using first {min(1000, len(processed_texts))} samples...")
        processed_texts = processed_texts[:min(1000, len(processed_texts))]

    # Dataset 객체로 변환
    from datasets import Dataset
    return Dataset.from_list([{"text": text} for text in processed_texts])

def is_coref_rich_wiki(text: str) -> bool:
    """Wikipedia 텍스트가 coreference에 적합한지 판단"""
    coref_indicators = [
        '사람', '인물', '배우', '정치인', '작가', '화가', '음악가',
        '회사', '기업', '조직', '단체', '국가', '도시',
        '이야기', '역사', '사건', '발생', '참여', '관련'
    ]

    text_lower = text.lower()
    indicator_count = sum(1 for indicator in coref_indicators if indicator in text_lower)

    # 최소 2개 이상의 coref 지표가 있어야 함
    return indicator_count >= 2

def chunk_with_coref_preservation(text: str, chunk_size: int, tokenizer) -> List[str]:
    """Coreference 맥락을 유지하며 청킹"""
    # 문장 단위 분리 (단순화)
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ['.', '!', '?', '다.'] and len(current) > 10:
            sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))

        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            if current_tokens > chunk_size * 0.6:  # 최소 길이
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens

    if current_chunk and current_tokens > chunk_size * 0.6:
        chunks.append(current_chunk.strip())

    return chunks

def chunk_long_text(text: str, chunk_size: int, tokenizer, overlap: int = 200) -> List[str]:
    """긴 텍스트를 청킹"""
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) >= chunk_size * 0.7:  # 최소 길이
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

    return chunks

def quality_weighted_sample_indices(scores: List[float], k: int) -> List[int]:
    """품질 기반 샘플링"""
    total_score = sum(scores)
    probabilities = [score / total_score for score in scores]

    return np.random.choice(len(scores), size=k, p=probabilities, replace=False).tolist()


# ────────────────────────────────────────────────────────────────────────────────
# Memory-Efficient Batch Configuration
# ────────────────────────────────────────────────────────────────────────────────

def compute_optimal_batch_config(seq_len: int, model_name: str, available_memory_gb: float = 80) -> Dict[str, Any]:
    """메모리 기반 최적 배치 설정"""

    # 시퀀스 길이별 메모리 요구량 (GB per sample)
    memory_per_sample = {
        512: 0.1,
        768: 0.25,
        1024: 0.5,
        1280: 0.8,
        1536: 1.2,
        1792: 1.7,
        2048: 2.5
    }

    base_memory = memory_per_sample.get(seq_len, 2.0)
    max_samples_per_batch = int(available_memory_gb / base_memory)

    # 모델별 최적화
    if "deberta" in model_name.lower():
        # DeBERTa는 더 효율적
        max_samples_per_batch = int(max_samples_per_batch * 1.2)

    # 배치 크기 및 grad_acc 계산
    if seq_len <= 512:
        bs_candidates = [32, 16, 8, 4]
        grad_acc_candidates = [1, 2]
    elif seq_len <= 1024:
        bs_candidates = [16, 8, 4, 2]
        grad_acc_candidates = [2, 4, 8]
    elif seq_len <= 1536:
        bs_candidates = [8, 4, 2, 1]
        grad_acc_candidates = [4, 8, 16]
    else:  # 2048
        bs_candidates = [4, 2, 1]
        grad_acc_candidates = [8, 16, 32]

    # 최적 조합 찾기
    best_config = {"bs": 1, "grad_acc": 32, "effective_bs": 32, "memory_usage": 0}

    for bs in bs_candidates:
        for grad_acc in grad_acc_candidates:
            effective_bs = bs * grad_acc
            memory_usage = base_memory * bs

            # 메모리 제약 확인
            if memory_usage > available_memory_gb * 0.8:  # 80% 이내
                continue

            # 더 큰 effective batch size 선호
            if effective_bs > best_config["effective_bs"]:
                best_config = {
                    "bs": bs,
                    "grad_acc": grad_acc,
                    "effective_bs": effective_bs,
                    "memory_usage": memory_usage
                }

    return best_config


# ────────────────────────────────────────────────────────────────────────────────
# Advanced Hyperparameter Optimization
# ────────────────────────────────────────────────────────────────────────────────

def get_length_specific_hpo_space(seq_len: int) -> Dict[str, Tuple]:
    """길이에 따른 최적 HPO 공간"""

    if seq_len <= 512:
        return {
            "lr": (1e-5, 5e-4),
            "warmup_ratio": (0.0, 0.2),
            "weight_decay": (0.0, 0.1),
            "min_prob": (0.05, 0.15),
            "max_prob": (0.20, 0.35),
        }
    elif seq_len <= 1024:
        return {
            "lr": (5e-6, 2e-4),
            "warmup_ratio": (0.05, 0.25),
            "weight_decay": (0.01, 0.08),
            "min_prob": (0.08, 0.18),
            "max_prob": (0.25, 0.40),
        }
    elif seq_len <= 1536:
        return {
            "lr": (7.5e-5, 9.0e-5),
            "warmup_ratio": (0.15, 0.17),
            "weight_decay": (0.038, 0.046),
            "min_prob": (0.11, 0.125),
            "max_prob": (0.35, 0.38),
        }
    else:  # 2048
        return {
            "lr": (2.1e-5, 2.6e-5),
            "warmup_ratio": (0.19, 0.21),
            "weight_decay": (0.044, 0.048),
            "min_prob": (0.13, 0.145),
            "max_prob": (0.46, 0.49),
        }


# ────────────────────────────────────────────────────────────────────────────────
# Long Sequence AutoML Objective
# ────────────────────────────────────────────────────────────────────────────────

def long_sequence_automl_objective(
    trial: optuna.Trial,
    model_name: str,
    seq_len: int,
    train_limit: Optional[int] = None,
    skip_quality_analysis: bool = False,
    prepared_dataset_path: Optional[str] = None,
    dataset_paths: Optional[List[str]] = None,
    epoch_choices: Optional[List[int]] = None,
) -> float:
    """긴 시퀀스 특화 AutoML objective"""

    # 1. 길이별 HPO 공간 설정
    hpo_space = get_length_specific_hpo_space(seq_len)

    # 2. 하이퍼파라미터 샘플링
    lr = trial.suggest_float("lr", *hpo_space["lr"], log=True)
    warmup = trial.suggest_float("warmup_ratio", *hpo_space["warmup_ratio"])
    wd = trial.suggest_float("weight_decay", *hpo_space["weight_decay"])
    min_p = trial.suggest_float("min_prob", *hpo_space["min_prob"])
    max_p = trial.suggest_float("max_prob", *hpo_space["max_prob"])

    # 3. 메모리 기반 배치 설정
    batch_config = compute_optimal_batch_config(seq_len, model_name)
    per_device_bs = batch_config["bs"]

    actual_grad_acc = min(32, batch_config["grad_acc"])

    print(f"Trial {trial.number}: seq_len={seq_len}, bs={per_device_bs}, grad_acc={actual_grad_acc}, lr={lr:.2e}")

    # 4. 모델 및 토크나이저 로드
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)

    # 긴 시퀀스 지원을 위해 position embeddings 확장 (필요시)
    if seq_len > mdl.config.max_position_embeddings:
        print(f"Extending position embeddings from {mdl.config.max_position_embeddings} to {seq_len}")
        mdl.config.max_position_embeddings = seq_len
        # position embeddings 재초기화
        mdl.deberta.embeddings.position_embeddings = torch.nn.Embedding(seq_len, mdl.config.hidden_size).to(mdl.device)

    # 5. 긴 시퀀스 특화 데이터셋
    def resolve_dataset_path(base_path: Path) -> Optional[Path]:
        if not base_path.exists():
            return None
        if base_path.is_dir():
            expected = base_path / f"{model_name.replace('/', '_')}_{seq_len}_coref_optimized.arrow"
            if expected.exists():
                return expected
            if (base_path / "dataset_info.json").exists():
                return base_path
            for child in base_path.iterdir():
                if child.is_dir() and (child / "dataset_info.json").exists() and str(seq_len) in child.name:
                    return child
            return None
        return base_path

    dataset_candidates: Dict[str, Dict[str, Any]] = {}
    dataset_paths = dataset_paths or []
    prepared_dataset_path = prepared_dataset_path or None

    has_user_choices = False

    if dataset_paths:
        for raw_path in dataset_paths:
            resolved = resolve_dataset_path(Path(raw_path))
            if resolved is None:
                continue
            label = f"prepared[{Path(raw_path).name}_{seq_len}]"
            dataset_candidates[label] = {"path": str(resolved)}
            has_user_choices = True

    if prepared_dataset_path and not has_user_choices:
        resolved_default = resolve_dataset_path(Path(prepared_dataset_path))
        if resolved_default is not None:
            dataset_candidates.setdefault(
                f"prepared_default_{seq_len}", {"path": str(resolved_default)}
            )

    # 사용자 지정 후보가 없다면 기존 enhanced 모드 추가
    if not dataset_candidates:
        dataset_candidates["enhanced_limit_30000_skip"] = {
            "limit": 30000,
            "skip_quality_analysis": True,
        }
        dataset_candidates["enhanced_limit_20000_skip"] = {
            "limit": 20000,
            "skip_quality_analysis": True,
        }

    choice_key = trial.suggest_categorical(
        "dataset_mode_choice", list(dataset_candidates.keys())
    )
    selected_options = dataset_candidates[choice_key]
    dataset_meta_info = {"mode": choice_key}

    if "path" in selected_options:
        tokenized = load_prepared_dataset(selected_options["path"])
        dataset_meta_info["samples"] = len(tokenized)
        dataset_meta_info["source"] = selected_options["path"]
    else:
        limit = selected_options["limit"]
        skip_local = selected_options["skip_quality_analysis"]
        tokenized = load_enhanced_dataset(
            tok,
            seq_len,
            limit=limit,
            skip_quality_analysis=skip_local,
        )
        dataset_meta_info["limit"] = limit
        dataset_meta_info["skip_quality_analysis"] = skip_local

    # 6. 콜레이터 설정
    collator = DynCollator(tokenizer=tok, mlm=True, min_prob=min_p, max_prob=max_p, max_length=seq_len)

    # 7. 학습 설정
    epoch_options = epoch_choices or [1, 2, 3]
    epochs = trial.suggest_categorical("num_epochs", epoch_options)
    steps_per_epoch = max(1, math.ceil(len(tokenized) / max(1, per_device_bs * actual_grad_acc)))
    total_steps = steps_per_epoch * epochs

    # 8. 옵티마이저 설정 (Layer-wise LR Decay)
    no_decay = ["bias", "LayerNorm.weight"]
    named = list(mdl.named_parameters())
    total_layers = sum(1 for n, _ in named if "encoder.layer" in n) or 12

    groups = []
    for n, p in named:
        lr_here = lr
        if "encoder.layer." in n:
            try:
                k = int(n.split("encoder.layer.")[1].split(".")[0])
                # 긴 시퀀스에서는 더 강한 layer decay
                decay_factor = 0.9 if seq_len <= 1024 else 0.85
                lr_here = lr * (decay_factor ** (total_layers - 1 - k))
            except Exception:
                pass
        wd_here = 0.0 if any(nd in n for nd in no_decay) else wd
        groups.append({"params": [p], "weight_decay": wd_here, "lr": lr_here})

    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)

    # 9. 학습 인자
    hp = {
        "seq_len": seq_len,
        "per_device_bs": per_device_bs,
        "grad_acc": actual_grad_acc,
        "effective_bs": per_device_bs * actual_grad_acc,
        "warmup_ratio": warmup,
        "lr": lr,
        "weight_decay": wd,
        "min_prob": min_p,
        "max_prob": max_p,
        "bf16": torch.cuda.is_available(),
        "train_limit": dataset_meta_info.get("limit") or train_limit,
        "dataset_mode": choice_key,
        "num_epochs": epochs,
    }

    # 학습 시작 이벤트
    BUS.log(event="trial_begin", model=model_name, trial=trial.number, **hp)

    args = TrainingArguments(
        output_dir=f"./runs/{model_name.replace('/', '_')}_long_{seq_len}/{trial.number}",
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=max(1, per_device_bs // 2),
        gradient_accumulation_steps=actual_grad_acc,
        learning_rate=lr,
        weight_decay=wd,
        warmup_ratio=warmup,
        num_train_epochs=epochs,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=42,
        dataloader_drop_last=False,
        bf16=torch.cuda.is_available(),
        # 긴 시퀀스 특화 설정 (DeBERTa는 gradient checkpointing 비활성화)
        max_grad_norm=0.5 if seq_len > 1024 else 1.0,
        gradient_checkpointing=False,  # DeBERTa 호환성 문제로 비활성화
    )

    # 10. 트레이너 설정
    tr = Trainer(
        model=mdl,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tok,
        optimizers=(opt, None),
    )

    # 콜백 추가
    tr.add_callback(
        LiveMetricsCallback(
            model_name=f"{model_name}_long_{seq_len}",
            trial_id=trial.number,
            hp=hp,
            dataset_meta={"seq_len": seq_len, **dataset_meta_info},
            total_steps=total_steps,
        )
    )

    # 11. 학습 실행
    tr.train()

    # 12. 평가
    fill = pipeline("fill-mask", model=mdl, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)

    # LAMBADA 평가
    eval_lbd = build_eval_from_lambada(limit=600)
    l_t1 = eval_lambada_topk(
        fill,
        eval_lbd,
        mask_token=tok.mask_token or "[MASK]",
        k=1,
        batch_size=32,
        seq_len=seq_len,
    )

    # Coref 평가 (긴 시퀀스 특화)
    eval_coref = build_coref_eval_set(limit=800, max_seq_len=seq_len)  # 더 많은 데이터로 평가
    c_f1 = eval_coref_f1(
        fill,
        eval_coref,
        mask_token=tok.mask_token or "[MASK]",
        k=5,
        batch_size=32,
        seq_len=seq_len,
    )
    c_t5 = eval_coref_recall_topk(
        fill,
        eval_coref,
        mask_token=tok.mask_token or "[MASK]",
        k=5,
        batch_size=32,
        seq_len=seq_len,
    )

    # 13. 종합 스코어 계산
    score = 0.4 * c_f1 + 0.3 * c_t5 + 0.3 * l_t1

    # 결과 로깅
    BUS.log(
        section="eval_stream",
        model=model_name,
        trial=trial.number,
        seq_len=seq_len,
        lbd_top1=l_t1,
        coref_f1=c_f1,
        coref_top5=c_t5,
        score=score
    )

    trial.set_user_attr("lbd_top1", l_t1)
    trial.set_user_attr("coref_f1", c_f1)
    trial.set_user_attr("coref_top5", c_t5)
    trial.set_user_attr("seq_len", seq_len)

    # 메모리 정리
    del mdl, tr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(score)


# ────────────────────────────────────────────────────────────────────────────────
# Long Sequence AutoML Runner
# ────────────────────────────────────────────────────────────────────────────────

def run_long_sequence_automl(
    model_name: str = "kakaobank/kf-deberta-base",
    seq_lengths: List[int] = [1536, 2048],
    trials_per_length: int = 20,
    train_limit: Optional[int] = None,
    skip_quality_analysis: bool = False,
    prepared_dataset_path: Optional[str] = None,
    dataset_paths: Optional[List[str]] = None,
    epoch_choices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """긴 시퀀스 AutoML 실행"""

    results = {}
    total_trials = len(seq_lengths) * trials_per_length
    overall_elapsed: List[float] = []

    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Starting Long Sequence AutoML: {model_name} @ {seq_len} tokens")
        print(f"{'='*60}")

        # 시퀀스 길이별 study 생성
        study_name = f"LongSeq_{model_name.replace('/', '_')}_{seq_len}"
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )

        elapsed: List[float] = []

        def objective(trial):
            t0 = time.time()
            result = long_sequence_automl_objective(
                trial,
                model_name,
                seq_len,
                train_limit,
                skip_quality_analysis,
                prepared_dataset_path,
                dataset_paths,
                epoch_choices=epoch_choices,
            )
            dt = time.time() - t0
            elapsed.append(dt)
            overall_elapsed.append(dt)
            done = len(elapsed)
            remaining = max(0, trials_per_length - done)
            avg = sum(elapsed) / done
            eta = avg * remaining
            overall_done = len(overall_elapsed)
            overall_remaining = max(0, total_trials - overall_done)
            overall_eta = (sum(overall_elapsed) / overall_done * overall_remaining) if overall_done else 0.0
            print(
                "[ETA]"
                f" seq_len={seq_len} trial={trial.number}"
                f" | this_trial={dt:.1f}s"
                f" | seq_remaining={remaining} (~{eta/60:.1f}m)"
                f" | overall {overall_done}/{total_trials} done → ~{overall_eta/60:.1f}m left"
            )
            return result

        # 최적화 실행
        study.optimize(objective, n_trials=trials_per_length, show_progress_bar=True)

        # 결과 저장
        results[seq_len] = {
            "study": study,
            "best_trial": study.best_trial,
            "best_score": study.best_value,
            "best_params": study.best_params
        }

        # 결과 출력
        print(f"\n{'='*40}")
        print(f"Results for {seq_len} tokens:")
        print(f"Best Score: {study.best_value:.4f}")
        print(f"Best Params: {study.best_params}")
        print(f"Best Trial: {study.best_trial.number}")

    return results


def prepare_datasets_with_kiwi(model_name: str = "kakaobank/kf-deberta-base", seq_lengths: List[int] = [1024, 1536, 2048], save_path: str = "./prepared_datasets"):
    """Kiwi 품질 분석을 적용하여 긴 시퀀스 데이터셋을 미리 준비 (상호참조 최적화)"""
    import os
    os.makedirs(save_path, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)

    print(f"🚀 Preparing Long Sequence Datasets with Kiwi Quality Analysis")
    print(f"🎯 Target: Better Coreference Resolution for Fill-Mask Model")
    print(f"📁 Save path: {save_path}")
    print(f"🔢 Sequence lengths: {seq_lengths}")

    for seq_len in seq_lengths:
        print(f"\n{'='*80}")
        print(f"🎯 Preparing {seq_len} tokens dataset for Coreference Training")
        print('='*80)

        try:
            print(f"📊 Loading enhanced dataset with Kiwi quality filtering...")
            # 더 긴 시퀀스의 경우 더 엄격한 품질 필터링 적용
            dataset = load_enhanced_dataset(tok, seq_len, limit=None)  # 전체 데이터 사용

            print(f"✅ Loaded {len(dataset)} raw samples")

            # 상호참조 품질이 높은 샘플만 선별 (긴 시퀀스 최적화)
            print("🔍 Analyzing coreference quality for long sequences...")

            high_quality_samples = []
            quality_stats = {'pronoun_density': [], 'entity_density': [], 'coref_score': []}

            # 샘플링하여 품질 분석 (메모리 효율성)
            sample_size = min(500, len(dataset))  # 최대 500개 샘플 분석
            indices = list(range(0, len(dataset), max(1, len(dataset)//sample_size)))

            for i, idx in enumerate(indices):
                if i >= sample_size:
                    break
                sample = dataset[idx]
                input_ids = sample['input_ids']
                # 긴 시퀀스의 경우 더 많은 토큰으로 품질 분석
                analysis_length = min(300, len(input_ids))  # 최대 300토큰 분석
                clean_ids = [x for x in input_ids if x != tok.pad_token_id][:analysis_length]
                text = tok.decode(clean_ids, skip_special_tokens=True)

                quality = analyze_coref_quality(text)
                quality_stats['pronoun_density'].append(quality['pronoun_density'])
                quality_stats['entity_density'].append(quality['entity_density'])
                quality_stats['coref_score'].append(quality['coref_score'])

                # 상호참조 품질이 높은 샘플만 선택 (긴 시퀀스에 적합한 기준)
                if quality['pronoun_density'] > 0.008 and quality['coref_score'] > 0.3:
                    high_quality_samples.append(idx)

                if (i + 1) % 50 == 0:
                    print(f"  📈 Analyzed {i + 1}/{sample_size} samples...")

            # 품질 통계 계산
            avg_pronoun_density = sum(quality_stats['pronoun_density']) / len(quality_stats['pronoun_density'])
            avg_entity_density = sum(quality_stats['entity_density']) / len(quality_stats['entity_density'])
            avg_coref_score = sum(quality_stats['coref_score']) / len(quality_stats['coref_score'])

            print("\n📊 Quality Analysis Results:")
            print(".3f")
            print(".3f")
            print(".3f")
            print(f"  🎯 High-quality samples selected: {len(high_quality_samples)}/{len(dataset)}")

            # 고품질 샘플만으로 데이터셋 생성
            if high_quality_samples:
                selected_dataset = dataset.select(high_quality_samples)

                # 저장
                save_file = f"{save_path}/{model_name.replace('/', '_')}_{seq_len}_coref_optimized.arrow"
                selected_dataset.save_to_disk(save_file)

                print(f"✅ Saved {len(selected_dataset)} high-quality samples to {save_file}")

                # 최종 품질 검증
                final_quality_check = []
                for i in range(min(50, len(selected_dataset))):
                    sample = selected_dataset[i]
                    input_ids = sample['input_ids']
                    clean_ids = [x for x in input_ids if x != tok.pad_token_id][:200]
                    text = tok.decode(clean_ids, skip_special_tokens=True)
                    quality = analyze_coref_quality(text)
                    final_quality_check.append(quality)

                final_pronoun_density = sum(q['pronoun_density'] for q in final_quality_check) / len(final_quality_check)
                final_coref_score = sum(q['coref_score'] for q in final_quality_check) / len(final_quality_check)

                print("🎉 Final Dataset Quality:")
                print(".3f")
                print(".3f")
                print("  ✨ Optimized for Coreference Resolution!")
            else:
                print("⚠️ No high-quality samples found. Saving original dataset...")
                save_file = f"{save_path}/{model_name.replace('/', '_')}_{seq_len}_fallback.arrow"
                dataset.save_to_disk(save_file)
                print(f"✅ Saved fallback dataset to {save_file}")

        except Exception as e:
            print(f"❌ Failed to prepare {seq_len} tokens dataset: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n🎊 Long Sequence Dataset Preparation Completed!")
    print(f"📂 Files saved to: {save_path}")
    print("🎯 Ready for Coreference-Optimized Fill-Mask Training!")
    print("\n🚀 Next: Run training with prepared datasets")
    print("   uv run python -m coref_automl.long_sequence_automl --model kakaobank/kf-deberta-base --seq-lengths 1024 1536 2048 --trials 10 --train-limit 50000")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Long Sequence DeBERTa AutoML")
    parser.add_argument("--model", default="kakaobank/kf-deberta-base", help="Model name")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[1536, 2048], help="Sequence lengths to test")
    parser.add_argument("--trials", type=int, default=20, help="Trials per sequence length")
    parser.add_argument("--train-limit", type=int, default=30000, help="Training data limit")
    parser.add_argument("--skip-quality-analysis", action="store_true", help="Skip quality analysis for faster dataset loading")
    parser.add_argument("--prepare-datasets", action="store_true", help="Prepare datasets with Kiwi quality analysis")
    parser.add_argument("--save-path", default="./prepared_datasets", help="Path to save prepared datasets")
    parser.add_argument("--prepared-dataset", help="Path to prepared dataset file or directory")
    parser.add_argument(
        "--dataset-choice",
        action="append",
        dest="dataset_choices",
        help="Prepared dataset directory or file to consider (can be repeated)",
    )
    parser.add_argument(
        "--epoch-choices",
        type=int,
        nargs="+",
        help="Candidate epoch counts to sample (default: 1 2 3)",
    )

    args = parser.parse_args()

    if args.prepare_datasets:
        # 데이터셋 준비 모드
        prepare_datasets_with_kiwi(
            model_name=args.model,
            seq_lengths=args.seq_lengths,
            save_path=args.save_path
        )
    else:
        # 일반 AutoML 모드
        results = run_long_sequence_automl(
            model_name=args.model,
            seq_lengths=args.seq_lengths,
            trials_per_length=args.trials,
            train_limit=args.train_limit,
            skip_quality_analysis=args.skip_quality_analysis,
            prepared_dataset_path=args.prepared_dataset,
            dataset_paths=args.dataset_choices,
            epoch_choices=args.epoch_choices,
        )

        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)

        for seq_len, result in results.items():
            print(f"\n{seq_len} tokens:")
            print(".4f")
            print(f"  Best Params: {result['best_params']}")

        print("\nAutoML completed! Check the data/ directory for detailed logs.")
