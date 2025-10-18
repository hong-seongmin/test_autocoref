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
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, disable_caching, load_from_disk
from transformers import AutoTokenizer
from kiwipiepy import Kiwi
from tqdm import tqdm
def _init_kiwi():
    global KIWI
    from kiwipiepy import Kiwi
    KIWI = Kiwi()  # 프로세스마다 독립 초기화



# 상대 import를 절대 import로 변경 (직접 실행 지원)
try:
    from .coref_utils import is_noun
    from .bus import BUS
except ImportError:
    # 직접 실행 시
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from coref_automl.coref_utils import is_noun
    from coref_automl.bus import BUS

# disable_caching()

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
    is_streaming: bool = False

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

        # ===== 새로운 8개 데이터셋 =====

        # 1. SKT KoBEST HellaSwag
        DatasetSource(
            name="kobest_hellaswag",
            source="skt/kobest_v1",
            subset="hellaswag",
            split="train",
            domain="hellaswag",
            quality_weight=0.7,
            description="HellaSwag 한국어 버전"
        ),

        # 2. HPLT Korean (대규모 웹 크롤) - 기본 50만 샘플로 제한
        DatasetSource(
            name="hplt_korean",
            source="HPLT/HPLT2.0_cleaned",
            subset="kor_Hang",
            split="train",
            domain="hplt_general",
            quality_weight=0.6,
            description="HPLT 대규모 한국어 웹 크롤 (50만 샘플)",
            is_streaming=True
        ),

        # 3. 번역된 미국 뉴스
        DatasetSource(
            name="translated_us_news",
            source="nmixx-fin/ko-trans-us_news_retrieval",
            subset=None,
            split="train",
            domain="translated_news",
            quality_weight=0.7,
            description="번역된 미국 뉴스"
        ),

        # 4. 금융 뉴스 요약
        DatasetSource(
            name="finance_news_summ",
            source="nmixx-fin/twice_kr_finance_news_summ",
            subset=None,
            split="train",
            domain="finance_news",
            quality_weight=0.9,
            description="금융 뉴스 요약"
        ),

        # 5. 경제 뉴스 BQA
        DatasetSource(
            name="economy_bqa",
            source="nmixx-fin/twice_kr_news_bqa_cls",
            subset=None,
            split="train",
            domain="economy_news",
            quality_weight=0.9,
            description="경제 뉴스 BQA"
        ),

        # 6. 금융 뉴스 감성
        DatasetSource(
            name="finance_sentiment",
            source="nmixx-fin/twice_kr_fin_news_sent_cls",
            subset=None,
            split="train",
            domain="finance_sentiment",
            quality_weight=0.7,
            description="금융 뉴스 감성 분석"
        ),

        # 7. 네이버 뉴스 (최고 품질)
        DatasetSource(
            name="naver_news_gen",
            source="dev7halo/naver-news-summarization-ko-with-gen",
            subset=None,
            split="train",
            domain="naver_news",
            quality_weight=1.0,
            description="네이버 뉴스 요약 (최고 품질)"
        ),

        # 8. AIR-Bench QA 뉴스
        DatasetSource(
            name="air_bench_news",
            source="AIR-Bench/qa_news_ko",
            subset=None,
            split="corpus_default",
            domain="qa_news",
            quality_weight=0.9,
            description="AIR-Bench QA 뉴스",
            is_streaming=True
        ),
    ]

@lru_cache(maxsize=4096)
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

def _analyze_batch(text_batch):
    # 배치 단위 분석 (기존 analyze_coref_quality_cached 재사용 가능)
    return [analyze_coref_quality_cached(t) for t in text_batch]

def batch_analyze_quality(texts: List[str], batch_size: int = 2000, max_workers: Optional[int] = None) -> List[Dict[str, float]]:
    if max_workers is None:
        max_workers = max(1, min(os.cpu_count() or 1, 16))  # 과섭스크립션 방지

    print(f"  📊 총 {len(texts):,}개 샘플 품질 분석 시작 (워커: {max_workers}개)", flush=True)

    CHUNK_SIZE = 20000
    results, submitted = [], 0
    overall_start = time.time()

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_kiwi) as ex:
        futures = []
        for c in range(0, len(texts), CHUNK_SIZE):
            chunk = texts[c:c+CHUNK_SIZE]
            for b in range(0, len(chunk), batch_size):
                futures.append(ex.submit(_analyze_batch, chunk[b:b+batch_size]))
                submitted += 1

        done = 0
        last = time.time()
        for f in as_completed(futures):
            results.extend(f.result())
            done += 1
            now = time.time()
            if done % 20 == 0 or now - last > 10:
                pct = 100.0 * done / submitted
                speed = (len(results) / max(1e-9, (now - overall_start)))
                eta = (len(texts) - len(results)) / max(1e-6, speed)
                print(f"    ⚡ 진행: {pct:.1f}% | {speed:.1f} 샘플/초 | ETA {eta/60:.1f}분", flush=True)
                last = now

    print(f"  ✅ 전체 품질 분석 완료: {len(results):,}/{len(texts):,}", flush=True)
    return results

def load_and_preprocess_dataset(source: DatasetSource, tokenizer, target_seq_len: int, limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """단일 데이터셋 로드 및 전처리 (스트리밍 지원)"""

    start_time = time.time()

    try:
        print(f"📥 Loading {source.name}: {source.description}")

        # 데이터 로드
        load_kwargs = {"split": source.split}
        if source.subset:
            load_kwargs["name"] = source.subset

        # 스트리밍 데이터셋 처리
        if source.is_streaming:
            load_kwargs["streaming"] = True
            dataset_stream = load_dataset(source.source, **load_kwargs)

            # 스트리밍: limit 있으면 제한, 없으면 iterator로 직접 처리
            if limit:
                # 제한이 있으면 메모리에 로드
                samples = []
                for i, sample in enumerate(dataset_stream):
                    if i >= limit:
                        break
                    samples.append(sample)
                from datasets import Dataset
                dataset = Dataset.from_list(samples)
                print(f"  ✅ Loaded {len(dataset)} raw samples (limited)")

                # 도메인별 전처리
                processed_texts = preprocess_domain_texts(dataset, source, tokenizer, target_seq_len, is_iterator=False)
            else:
                # limit 없음: 대형 데이터셋은 기본 제한 적용 (50만 샘플)
                default_streaming_limit = 500000
                print(f"  🔄 스트리밍 모드 (최대 {default_streaming_limit:,}개)")

                # 제한된 iterator로 전처리
                samples = []
                for i, sample in enumerate(dataset_stream):
                    if i >= default_streaming_limit:
                        break
                    samples.append(sample)

                    # 1만개마다 진행 상황 출력
                    if (i + 1) % 10000 == 0:
                        print(f"\r  ⏳ 로딩 중: {i+1:,}/{default_streaming_limit:,} ({(i+1)/default_streaming_limit*100:.1f}%)", end="", flush=True)

                print(f"\r  ✅ 로딩 완료: {len(samples):,} 샘플" + " " * 20)

                from datasets import Dataset
                dataset = Dataset.from_list(samples)

                # 전처리
                processed_texts = preprocess_domain_texts(dataset, source, tokenizer, target_seq_len, is_iterator=False)
        else:
            dataset = load_dataset(source.source, **load_kwargs)

            # 샘플 제한
            if limit and limit < len(dataset):
                dataset = dataset.select(range(limit))

            print(f"  ✅ Loaded {len(dataset)} raw samples")

            # 도메인별 전처리
            processed_texts = preprocess_domain_texts(dataset, source, tokenizer, target_seq_len, is_iterator=False)

        if not processed_texts:
            print(f"  ⚠️ No valid texts after preprocessing")
            return None

        elapsed_time = time.time() - start_time
        print(f"  ✅ Preprocessed {len(processed_texts)} texts")
        print(f"  ⏱️  소요 시간: {elapsed_time:.1f}초 ({len(processed_texts)/elapsed_time:.1f} 샘플/초)")

        # Dataset 객체로 변환
        from datasets import Dataset
        return Dataset.from_list([{"text": text} for text in processed_texts])

    except Exception as e:
        print(f"  ❌ Failed to load {source.name}: {e}")
        return None

def preprocess_domain_texts(dataset, source: DatasetSource, tokenizer, target_seq_len: int, is_iterator: bool = False) -> List[str]:
    """도메인별 텍스트 전처리 (iterator 지원)"""

    processed_texts = []
    skipped = 0
    chunked = 0

    # iterator인 경우에도 진행률 표시 (샘플 카운트만)
    if is_iterator:
        print(f"  📝 스트리밍 전처리 중... (진행 중)", end="", flush=True)
        iterator = dataset
        show_progress = True
        progress_interval = 1000  # 1000개마다 진행 상황 출력
    else:
        iterator = tqdm(dataset, desc=f"  📝 전처리", unit="샘플", leave=False)
        show_progress = False

    sample_count = 0
    for item in iterator:
        sample_count += 1

        # 스트리밍 모드: 1000개마다 진행 상황 출력
        if show_progress and sample_count % progress_interval == 0:
            print(f"\r  📝 스트리밍 전처리 중... (처리: {sample_count:,}개, 생성: {len(processed_texts):,}개, 스킵: {skipped:,}개)", end="", flush=True)
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

            # ===== 새로운 8개 도메인 처리 =====
            elif source.domain == "hellaswag":
                text = item.get("paragraph", "").strip()
            elif source.domain == "hplt_general":
                text = item.get("text", "").strip()
            elif source.domain == "translated_news":
                text = item.get("trans_text", "").strip()
            elif source.domain == "finance_news":
                text = item.get("text", "").strip()
            elif source.domain == "economy_news":
                text = item.get("text", "").strip()
            elif source.domain == "finance_sentiment":
                text = item.get("text", "").strip()
            elif source.domain == "naver_news":
                text = item.get("document", "").strip()
            elif source.domain == "qa_news":
                text = item.get("text", "").strip()

            else:
                text = item.get("text", "").strip()

            # 기본 필터링
            if not text or len(text) < 50:
                skipped += 1
                continue

            # 토큰 길이 필터링 (강화)
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            token_len = len(tokens)

            # 최소 길이 체크 (Combined 기준: 100 토큰)
            # ★★★ 개선: 0.3x 제약(460토큰) → 100토큰으로 완화 (짧은 데이터셋 포함)
            if token_len < 100:
                skipped += 1
                continue

            # 적절한 길이: 그대로 사용
            if token_len <= target_seq_len:
                processed_texts.append(text)
            # 약간 긴 경우 (1.0 ~ 1.5배): truncate
            elif token_len <= target_seq_len * 1.5:
                truncated_ids = tokens[:target_seq_len]
                truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
                processed_texts.append(truncated_text)
            # 많이 긴 경우 (1.5배 초과): 청킹
            else:
                chunks = chunk_text_with_coref_preservation(text, target_seq_len, tokenizer)
                processed_texts.extend(chunks)
                chunked += 1

        except Exception as e:
            skipped += 1
            continue

    # 최종 진행 상황 출력
    if show_progress:
        print(f"\r  📊 전처리 완료: {len(processed_texts):,}개 생성 (처리: {sample_count:,}개, 스킵: {skipped:,}, 청킹: {chunked})                    ")
    else:
        print(f"  📊 전처리 완료: {len(processed_texts)}개 생성 (스킵: {skipped}, 청킹: {chunked})")

    return processed_texts

def chunk_text_with_coref_preservation(text: str, chunk_size: int, tokenizer, overlap: int = 200) -> List[str]:
    """Coreference 맥락을 유지하며 긴 텍스트 청킹 (길이 보장)"""

    # 문장 단위 분리
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ['.', '!', '?'] and len(current) > 10:
            sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False, truncation=False))

        # 문장 자체가 너무 길면 강제 truncate
        if sentence_tokens > chunk_size:
            # 현재 청크를 먼저 저장
            if current_chunk and current_tokens > chunk_size * 0.3:
                chunks.append(current_chunk.strip())

            # 긴 문장을 토큰 단위로 truncate
            sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
            truncated_ids = sentence_ids[:chunk_size]
            truncated_sentence = tokenizer.decode(truncated_ids, skip_special_tokens=True)
            chunks.append(truncated_sentence)

            # 청크 초기화
            current_chunk = ""
            current_tokens = 0
            continue

        # 일반 청킹 로직
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            if current_tokens > chunk_size * 0.3:  # 최소 길이
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens

    # 마지막 청크
    if current_chunk and current_tokens > chunk_size * 0.3:
        chunks.append(current_chunk.strip())

    # 최종 검증: 모든 청크가 chunk_size 이하인지 확인
    verified_chunks = []
    for chunk in chunks:
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False, truncation=False))
        if chunk_tokens <= chunk_size:
            verified_chunks.append(chunk)
        else:
            # 여전히 길면 강제 truncate
            chunk_ids = tokenizer.encode(chunk, add_special_tokens=False)
            truncated_ids = chunk_ids[:chunk_size]
            truncated_chunk = tokenizer.decode(truncated_ids, skip_special_tokens=True)
            verified_chunks.append(truncated_chunk)

    return verified_chunks

def prepare_coref_optimized_datasets(
    model_name: str = "kakaobank/kf-deberta-base",
    seq_lengths: List[int] = [1024, 1536, 2048],
    save_path: str = "./prepared_datasets",
    quality_threshold: float = 0.3,  # ★★★ 0.6 → 0.3 완화 (더 많은 데이터 포함)
    max_samples_per_length: int = None
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

        # 토크나이저 max_length 동적 확장
        original_max_length = tokenizer.model_max_length
        tokenizer.model_max_length = seq_len
        print(f"   Tokenizer max_length: {original_max_length} → {seq_len}")

        all_datasets = []
        total_samples = 0

        # 각 데이터 소스에서 데이터 수집
        for source in sources:
            try:
                print(f"\n📥 처리 중: {source.name}")

                # 데이터 로드 및 전처리 (limit 계산: None이면 무제한)
                limit_per_source = max_samples_per_length // len(sources) if max_samples_per_length else None
                dataset = load_and_preprocess_dataset(source, tokenizer, seq_len, limit=limit_per_source)

                if dataset is None or len(dataset) == 0:
                    continue

                print(f"  🔍 품질 분석 중... ({len(dataset)} 샘플)")

                # ★★★ MLM v3 개선: 샘플링 제거 - 전체 데이터 품질 분석 ★★★
                # 모든 샘플 분석 (병렬 처리로 충분히 빠름)
                print(f"  ⚡ 전체 샘플 품질 분석 (샘플링 없음)")

                # 텍스트 추출 (전체)
                texts = [dataset[idx]['text'] for idx in range(len(dataset))]

                # 병렬 품질 분석 (전체)
                qualities = batch_analyze_quality(texts, batch_size=500, max_workers=None)

                # 인덱스도 전체
                indices = list(range(len(dataset)))

                # 품질 필터링
                high_quality_samples = []
                quality_stats = {'analyzed': len(qualities), 'passed': 0, 'avg_quality': 0}

                for idx, quality in zip(indices, qualities):
                    quality_stats['avg_quality'] += quality['quality_score']
                    if quality['quality_score'] >= quality_threshold:
                        high_quality_samples.append(idx)
                        quality_stats['passed'] += 1

                quality_stats['avg_quality'] /= max(1, quality_stats['analyzed'])
                pass_rate = (quality_stats['passed'] / max(1, quality_stats['analyzed'])) * 100

                print(f"  ✅ 품질 분석 완료: {quality_stats['passed']}/{quality_stats['analyzed']} 통과 ({pass_rate:.1f}%)")
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

            # 최종 샘플 제한 (max_samples_per_length가 None이 아닐 때만)
            if max_samples_per_length and len(combined_dataset) > max_samples_per_length:
                combined_dataset = combined_dataset.select(range(max_samples_per_length))

            # 토큰화
            print(f"🔤 토큰화 중... ({len(combined_dataset)} 샘플)")

            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=seq_len,
                    # return_tensors="pt"
                )

            # 시퀀스 길이에 따라 배치 크기와 프로세스 수 동적 조정 (메모리 효율)
            if seq_len >= 2048:
                batch_size = 100
                num_proc = 4
                print(f"  ⚙️  토큰화 설정 (긴 시퀀스): batch_size={batch_size}, num_proc={num_proc}")
            elif seq_len >= 1536:
                batch_size = 200
                num_proc = 8
                print(f"  ⚙️  토큰화 설정 (중간 시퀀스): batch_size={batch_size}, num_proc={num_proc}")
            else:
                batch_size = 500
                num_proc = 12
                print(f"  ⚙️  토큰화 설정 (짧은 시퀀스): batch_size={batch_size}, num_proc={num_proc}")

            def safe_map(ds, fn, **kwargs):
                try:
                    return ds.map(fn, **kwargs)
                except BrokenPipeError:
                    # 워커 줄여서 1회 재시도
                    kwargs["num_proc"] = 1
                    return ds.map(fn, **kwargs)

            N_SHARDS = 16
            shard_paths = []

            for i in range(N_SHARDS):
                shard = combined_dataset.shard(num_shards=N_SHARDS, index=i, contiguous=True)
                tok = safe_map(
                    shard,
                    tokenize_function,
                    batched=True,
                    batch_size=batch_size,
                    remove_columns=["text"],
                    num_proc=num_proc,
                    writer_batch_size=1000,
                    load_from_cache_file=True,
                    desc=f"Tokenizing shard {i+1}/{N_SHARDS} @ {seq_len}"
                )
                out_dir = f"{save_path}/tmp_{seq_len}_shard_{i}"
                tok.save_to_disk(out_dir)
                shard_paths.append(out_dir)

            tokenized_dataset = concatenate_datasets([load_from_disk(p) for p in shard_paths])
            tokenized_dataset.save_to_disk( f"{save_path}/{model_name.replace('/', '_')}_{seq_len}_coref_optimized")


            # print(f"✅ 저장 완료: {save_file}")
            print(f"📊 최종 데이터셋: {len(tokenized_dataset)} 샘플 × {seq_len} 토큰")

            # 최종 품질 검증
            print("🎯 최종 품질 검증 중...")

            def preview_text_from_ids(input_ids, tokenizer, max_tokens=300):
                from itertools import compress
                out = tokenizer.prepare_for_model(input_ids, add_special_tokens=False)
                ids = input_ids[:max_tokens]
                return tokenizer.decode(ids, skip_special_tokens=True)

            final_qualities = []
            for i in range(min(100, len(tokenized_dataset))):
                sample = tokenized_dataset[i]
                input_ids = sample['input_ids']
                clean_ids = [x for x in input_ids if x != tokenizer.pad_token_id][:300]
                ids = tokenized_dataset[i]["input_ids"]
                text = preview_text_from_ids(ids, tokenizer, 300)
                quality = analyze_coref_quality(text)
                final_qualities.append(quality)

            avg_final_quality = sum(q['quality_score'] for q in final_qualities) / len(final_qualities)
            avg_pronoun_density = sum(q['pronoun_density'] for q in final_qualities) / len(final_qualities)
            avg_entity_density = sum(q['entity_density'] for q in final_qualities) / len(final_qualities)

            print("📊 최종 품질 지표:")
            print(f"  📈 평균 품질 점수: {avg_final_quality:.3f}")
            print(f"  🔤 대명사 밀도: {avg_pronoun_density:.3f}")
            print(f"  🏢 개체 밀도: {avg_entity_density:.3f}")

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
    parser.add_argument("--save-path", default="./prepared_datasets_mlm_v2", help="저장 경로")
    parser.add_argument("--quality-threshold", type=float, default=0.6, help="품질 임계값")
    parser.add_argument("--max-samples", type=lambda x: None if x.lower() == 'none' else int(x), default=None, help="최대 샘플 수 (None=무제한)")

    args = parser.parse_args()

    prepare_coref_optimized_datasets(
        model_name=args.model,
        seq_lengths=args.seq_lengths,
        save_path=args.save_path,
        quality_threshold=args.quality_threshold,
        max_samples_per_length=args.max_samples
    )