# coref_automl/dataset_preparation.py
"""
고품질 Coreference 데이터셋 사전 준비 시스템
- 긴 시퀀스(1024~2048) 지원
- Kiwi 품질 분석 기반 필터링
- 상호참조 최적화
"""

from __future__ import annotations
import os
import gc
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, disable_caching
from transformers import AutoTokenizer
from kiwipiepy import Kiwi

from .coref_utils import is_noun
from .bus import BUS

disable_caching()

# Kiwi 글로벌 인스턴스
KIWI = Kiwi()

@dataclass
class DatasetSource:
    """데이터셋 소스 설정"""
    name: str
    source: str
    subset: Optional[str]
    split: str
    domain: str
    quality_weight: float
    description: str

def get_all_dataset_sources() -> List[DatasetSource]:
    """모든 사용 가능한 데이터셋 소스"""
    return [
        # Wikipedia - 긴 문서, 다양한 주제
        DatasetSource(
            name="wiki_ko",
            source="wikimedia/wikipedia",
            subset="20231101.ko",
            split="train",
            domain="encyclopedia",
            quality_weight=0.8,
            description="한국어 위키백과 - 다양한 주제의 긴 문서"
        ),

        # KLUE MRC - 질문-답변, 긴 맥락
        DatasetSource(
            name="klue_mrc",
            source="klue",
            subset="mrc",
            split="train",
            domain="qa_long",
            quality_weight=1.0,
            description="KLUE MRC - 긴 맥락의 질문-답변 데이터"
        ),

        # KLUE YNAT - 뉴스 기사
        DatasetSource(
            name="klue_ynat",
            source="klue",
            subset="ynat",
            split="train",
            domain="news_topic",
            quality_weight=0.9,
            description="KLUE YNAT - 뉴스 기사 제목과 내용"
        ),

        # KorQuAD - 질문-답변
        DatasetSource(
            name="korquad",
            source="squad_kor_v1",
            subset=None,
            split="train",
            domain="qa_general",
            quality_weight=0.7,
            description="KorQuAD - 한국어 질문-답변 데이터셋"
        ),

        # KLUE STS - 문장 유사도 (긴 텍스트 조합용)
        DatasetSource(
            name="klue_sts",
            source="klue",
            subset="sts",
            split="train",
            domain="similarity",
            quality_weight=0.6,
            description="KLUE STS - 문장 유사도 (긴 텍스트 생성용)"
        ),

        # KLUE NLI - 추론 (긴 텍스트 조합용)
        DatasetSource(
            name="klue_nli",
            source="klue",
            subset="nli",
            split="train",
            domain="inference",
            quality_weight=0.5,
            description="KLUE NLI - 자연어 추론 (긴 텍스트 생성용)"
        ),
    ]

@lru_cache(maxsize=10000)
def analyze_coref_quality_cached(text: str) -> Dict[str, float]:
    """Kiwi를 사용한 고급 Coreference 품질 분석 (캐싱 적용)"""

    # 형태소 분석
    tokens = KIWI.tokenize(text)

    # 대명사와 개체 분석
    pronouns = []
    entities = []
    verbs = []
    meaningful_words = []

    for token in tokens:
        # Kiwi 태그 설명:
        # NP: 대명사 (그, 그녀, 이것 등)
        # NNG/NNP/NNB: 일반명사/고유명사/의존명사
        # VV/VX/VA: 동사/보조동사/형용사
        if token.tag == 'NP':  # 대명사
            pronouns.append(token.form)
        elif token.tag in ['NNG', 'NNP', 'NNB']:  # 명사류
            if len(token.form) > 1:  # 한 글자 제외
                entities.append(token.form)
        elif token.tag.startswith(('V', 'VA')):  # 동사/형용사
            verbs.append(token.form)

        # 의미 있는 단어 수 계산 (명사, 동사, 형용사 등)
        if token.tag.startswith(('N', 'V', 'M', 'VA', 'VV')):
            meaningful_words.append(token.form)

    # 통계 계산
    pronoun_count = len(pronouns)
    entity_count = len(entities)
    verb_count = len(verbs)
    total_words = len(meaningful_words)

    # 밀도 계산
    pronoun_density = pronoun_count / max(1, total_words)
    entity_density = entity_count / max(1, total_words)
    verb_density = verb_count / max(1, total_words)

    # 고급 Coreference 점수 계산
    # 1. 대명사-개체 상호작용 점수
    coref_interaction = min(1.0, (pronoun_density * 20) + (entity_density * 3) + (pronoun_density * entity_density * 50))

    # 2. 텍스트 복잡도 점수 (다양한 품사 사용)
    complexity_score = min(1.0, (entity_density * 10) + (verb_density * 5) + (pronoun_density * 15))

    # 3. Coreference 적합성 점수 (대명사와 개체의 균형)
    balance_score = 1.0 - abs(pronoun_density - entity_density * 0.3)  # 이상적인 비율

    # 종합 품질 점수
    quality_score = (coref_interaction * 0.5) + (complexity_score * 0.3) + (balance_score * 0.2)

    return {
        'pronoun_density': pronoun_density,
        'entity_density': entity_density,
        'verb_density': verb_density,
        'coref_score': coref_interaction,
        'complexity_score': complexity_score,
        'balance_score': balance_score,
        'quality_score': quality_score,
        'pronoun_count': pronoun_count,
        'entity_count': entity_count,
        'verb_count': verb_count,
        'total_words': total_words,
        'unique_pronouns': len(set(pronouns)),
        'unique_entities': len(set(entities)),
    }

def analyze_coref_quality(text: str) -> Dict[str, float]:
    """캐싱된 품질 분석 함수"""
    return analyze_coref_quality_cached(text)

def batch_analyze_quality(texts: List[str], batch_size: int = 100, max_workers: Optional[int] = None) -> List[Dict[str, float]]:
    """배치 단위 고성능 품질 분석"""
    if max_workers is None:
        max_workers = min(16, multiprocessing.cpu_count())  # 최대 16개 워커

    results = []

    # 배치로 나누어 처리
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(analyze_coref_quality_cached, batch_texts))
            results.extend(batch_results)

    return results

def load_and_preprocess_dataset(source: DatasetSource, tokenizer, target_seq_len: int, limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """단일 데이터셋 로드 및 전처리"""

    try:
        print(f"📥 Loading {source.name}: {source.description}")

        # 데이터 로드
        load_kwargs = {"split": source.split}
        if source.subset:
            load_kwargs["name"] = source.subset

        dataset = load_dataset(source.source, **load_kwargs)

        # 샘플 제한
        if limit and limit < len(dataset):
            dataset = dataset.select(range(limit))

        print(f"  ✅ Loaded {len(dataset)} raw samples")

        # 도메인별 전처리
        processed_texts = preprocess_domain_texts(dataset, source, tokenizer, target_seq_len)

        if not processed_texts:
            print(f"  ⚠️ No valid texts after preprocessing")
            return None

        print(f"  ✅ Preprocessed {len(processed_texts)} texts")

        # Dataset 객체로 변환
        from datasets import Dataset
        return Dataset.from_list([{"text": text} for text in processed_texts])

    except Exception as e:
        print(f"  ❌ Failed to load {source.name}: {e}")
        return None

def preprocess_domain_texts(dataset, source: DatasetSource, tokenizer, target_seq_len: int) -> List[str]:
    """도메인별 텍스트 전처리"""

    processed_texts = []

    for item in dataset:
        try:
            # 도메인별 텍스트 추출
            if source.domain == "encyclopedia":
                text = item.get("text", "").strip()
            elif source.domain == "qa_long":
                context = item.get("context", "")
                question = item.get("question", "")
                answer = item.get("answers", {}).get("text", [""])[0] if item.get("answers") else ""
                text = f"{context} {question} {answer}".strip()
            elif source.domain == "news_topic":
                title = item.get("title", "")
                content = item.get("content", item.get("description", ""))
                text = f"{title} {content}".strip()
            elif source.domain == "qa_general":
                context = item.get("context", "")
                question = item.get("question", "")
                answer = item.get("answers", {}).get("text", [""])[0] if item.get("answers") else ""
                text = f"{context} {question} {answer}".strip()
            elif source.domain == "similarity":
                sentence1 = item.get("sentence1", item.get("text", ""))
                sentence2 = item.get("sentence2", "")
                text = f"{sentence1} {sentence2}".strip()
            elif source.domain == "inference":
                premise = item.get("premise", "")
                hypothesis = item.get("hypothesis", "")
                if premise and hypothesis:
                    text = f"{premise} {hypothesis}".strip()
                else:
                    continue
            else:
                text = item.get("text", "").strip()

            # 기본 필터링
            if not text or len(text) < 50:
                continue

            # 토큰 길이 필터링 (긴 시퀀스 대비)
            tokens = tokenizer.encode(text)
            if len(tokens) < target_seq_len * 0.3:  # 최소 길이
                continue
            if len(tokens) > target_seq_len * 2.0:  # 최대 길이 (청킹 고려)
                # 긴 텍스트 청킹
                chunks = chunk_text_with_coref_preservation(text, target_seq_len, tokenizer)
                processed_texts.extend(chunks)
            else:
                processed_texts.append(text)

        except Exception as e:
            continue

    return processed_texts

def chunk_text_with_coref_preservation(text: str, chunk_size: int, tokenizer, overlap: int = 200) -> List[str]:
    """Coreference 맥락을 유지하며 긴 텍스트 청킹"""

    # 문장 단위 분리
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

def prepare_coref_optimized_datasets(
    model_name: str = "kakaobank/kf-deberta-base",
    seq_lengths: List[int] = [1024, 1536, 2048],
    save_path: str = "./prepared_datasets",
    quality_threshold: float = 0.6,
    max_samples_per_length: int = 50000
):
    """
    상호참조 최적화된 데이터셋 사전 준비
    - Kiwi 품질 분석 기반 필터링
    - 긴 시퀀스 지원
    - 메모리 효율적 처리
    """

    import os
    os.makedirs(save_path, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("🚀 고품질 Coreference 데이터셋 사전 준비 시스템")
    print(f"🎯 모델: {model_name}")
    print(f"📏 시퀀스 길이: {seq_lengths}")
    print(f"⭐ 품질 임계값: {quality_threshold}")
    print(f"💾 저장 경로: {save_path}")
    print("=" * 80)

    sources = get_all_dataset_sources()

    for seq_len in seq_lengths:
        print(f"\n🎯 {seq_len} 토큰 데이터셋 준비 중...")

        all_datasets = []
        total_samples = 0

        # 각 데이터 소스에서 데이터 수집
        for source in sources:
            try:
                print(f"\n📥 처리 중: {source.name}")

                # 데이터 로드 및 전처리
                dataset = load_and_preprocess_dataset(source, tokenizer, seq_len, limit=max_samples_per_length // len(sources))

                if dataset is None or len(dataset) == 0:
                    continue

                print(f"  🔍 품질 분석 중... ({len(dataset)} 샘플)")

                # 품질 분석 및 필터링
                high_quality_samples = []
                quality_stats = {'analyzed': 0, 'passed': 0, 'avg_quality': 0}

                # 샘플링하여 분석 (메모리 효율성)
                sample_size = min(1000, len(dataset))
                indices = list(range(0, len(dataset), max(1, len(dataset) // sample_size)))

                for idx in indices:
                    sample = dataset[idx]
                    text = sample['text']

                    quality = analyze_coref_quality(text)
                    quality_stats['analyzed'] += 1
                    quality_stats['avg_quality'] += quality['quality_score']

                    # 품질 필터링
                    if quality['quality_score'] >= quality_threshold:
                        high_quality_samples.append(idx)
                        quality_stats['passed'] += 1

                quality_stats['avg_quality'] /= max(1, quality_stats['analyzed'])

                print(f"  ✅ 품질 분석 완료: {quality_stats['passed']}/{quality_stats['analyzed']} 통과")
                print(f"    📊 평균 품질 점수: {quality_stats['avg_quality']:.3f}")
                # 고품질 샘플만 선택
                if high_quality_samples:
                    filtered_dataset = dataset.select(high_quality_samples)
                    all_datasets.append(filtered_dataset)
                    total_samples += len(filtered_dataset)

                    print(f"  🎯 선택된 샘플: {len(filtered_dataset)}")

            except Exception as e:
                print(f"  ❌ {source.name} 처리 실패: {e}")
                continue

        # 데이터 통합
        if all_datasets:
            print(f"\n🔄 {seq_len} 토큰 데이터셋 통합 중... ({total_samples} 샘플)")

            combined_dataset = concatenate_datasets(all_datasets)
            combined_dataset = combined_dataset.shuffle(seed=42)

            # 최종 샘플 제한
            if len(combined_dataset) > max_samples_per_length:
                combined_dataset = combined_dataset.select(range(max_samples_per_length))

            # 토큰화
            print(f"🔤 토큰화 중... ({len(combined_dataset)} 샘플)")

            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=seq_len,
                    return_tensors="pt"
                )

            tokenized_dataset = combined_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                remove_columns=["text"],
                num_proc=min(16, multiprocessing.cpu_count())
            )

            # 저장
            save_file = f"{save_path}/{model_name.replace('/', '_')}_{seq_len}_coref_optimized.arrow"
            tokenized_dataset.save_to_disk(save_file)

            print(f"✅ 저장 완료: {save_file}")
            print(f"📊 최종 데이터셋: {len(tokenized_dataset)} 샘플 × {seq_len} 토큰")

            # 최종 품질 검증
            print("🎯 최종 품질 검증 중...")

            final_qualities = []
            for i in range(min(100, len(tokenized_dataset))):
                sample = tokenized_dataset[i]
                input_ids = sample['input_ids']
                clean_ids = [x for x in input_ids if x != tokenizer.pad_token_id][:300]
                text = tokenizer.decode(clean_ids, skip_special_tokens=True)
                quality = analyze_coref_quality(text)
                final_qualities.append(quality)

            avg_final_quality = sum(q['quality_score'] for q in final_qualities) / len(final_qualities)
            avg_pronoun_density = sum(q['pronoun_density'] for q in final_qualities) / len(final_qualities)
            avg_entity_density = sum(q['entity_density'] for q in final_qualities) / len(final_qualities)

            print("📊 최종 품질 지표:")
            print(".3f")
            print(".3f")
            print(".3f")

            rating = '⭐ 우수' if avg_final_quality > 0.8 else '✅ 양호' if avg_final_quality > 0.6 else '⚠️ 보통'
            print(f"🏆 품질 등급: {rating}")

        else:
            print(f"⚠️ {seq_len} 토큰 데이터셋 생성 실패")

    print("\n🎊 모든 데이터셋 준비 완료!")
    print(f"📂 저장 위치: {save_path}")
    print("🚀 이제 학습을 시작할 수 있습니다!")

if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description="고품질 Coreference 데이터셋 사전 준비")
    parser.add_argument("--model", default="kakaobank/kf-deberta-base", help="모델 이름")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[1024, 1536, 2048], help="시퀀스 길이들")
    parser.add_argument("--save-path", default="./prepared_datasets", help="저장 경로")
    parser.add_argument("--quality-threshold", type=float, default=0.6, help="품질 임계값")
    parser.add_argument("--max-samples", type=int, default=50000, help="최대 샘플 수")

    args = parser.parse_args()

    prepare_coref_optimized_datasets(
        model_name=args.model,
        seq_lengths=args.seq_lengths,
        save_path=args.save_path,
        quality_threshold=args.quality_threshold,
        max_samples_per_length=args.max_samples
    )