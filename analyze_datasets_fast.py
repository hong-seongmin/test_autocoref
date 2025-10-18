#!/usr/bin/env python3
"""
HuggingFace 데이터셋 8개 분석 스크립트 (최적화 버전)
Entity coreference 훈련 적합성 평가
"""

import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from kiwipiepy import Kiwi
import numpy as np
from collections import Counter
import re
from tqdm import tqdm

# Kiwi 초기화
print("Kiwi 초기화 중...")
kiwi = Kiwi()

def analyze_nnp_repetition(text, kiwi_instance):
    """고유명사(NNP) 반복 패턴 분석"""
    try:
        result = kiwi_instance.analyze(text)
        tokens = result[0][0] if result else []

        nnp_tokens = [token.form for token in tokens if token.tag == 'NNP' and len(token.form) > 1]

        if len(nnp_tokens) < 2:
            return False, []

        nnp_counter = Counter(nnp_tokens)
        repeated = [entity for entity, count in nnp_counter.items() if count >= 2]

        return len(repeated) > 0, repeated
    except:
        return False, []

def count_pronouns(text, kiwi_instance):
    """대명사 개수 세기"""
    try:
        result = kiwi_instance.analyze(text)
        tokens = result[0][0] if result else []
        pronouns = [token.form for token in tokens if token.tag in ['NP', 'NNP']]
        return len([p for p in pronouns if p in ['그', '그녀', '그것', '이', '저', '이것', '저것', '그들', '이들']])
    except:
        return 0

def analyze_dataset(dataset_name, subset=None, split='train', text_field='text', sample_size=500, use_streaming=False):
    """데이터셋 분석"""
    print(f"\n{'='*80}")
    print(f"데이터셋: {dataset_name}" + (f" (subset: {subset})" if subset else ""))
    print(f"텍스트 필드: {text_field}")
    print(f"{'='*80}")

    try:
        # 데이터셋 로드
        print("데이터셋 로딩 중...")

        if use_streaming:
            # Streaming 모드
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=True)

            # 스트리밍에서 샘플 추출
            texts = []
            total_samples = 0
            for i, item in enumerate(dataset):
                if i >= sample_size:
                    break
                try:
                    text = item[text_field]
                    if text and isinstance(text, str):
                        texts.append(text.strip())
                except:
                    pass

            total_samples = f"~{sample_size * 100:,}+ (streaming)"

        else:
            # 일반 모드
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)

            total_samples = len(dataset)
            print(f"✓ 전체 샘플 수: {total_samples:,}")

            # 샘플링
            sample_indices = np.random.choice(min(total_samples, sample_size), min(sample_size, total_samples), replace=False)

            # 텍스트 추출
            texts = []
            for idx in sample_indices:
                try:
                    text = dataset[int(idx)][text_field]
                    if text and isinstance(text, str):
                        texts.append(text.strip())
                except:
                    continue

        if not texts:
            print("✗ 텍스트 추출 실패")
            return None

        print(f"✓ 분석할 샘플 수: {len(texts):,}")

        # 샘플 예시 출력
        print(f"\n[샘플 예시] (처음 3개)")
        for i, text in enumerate(texts[:3]):
            print(f"\n샘플 {i+1}:")
            print("-" * 80)
            print(text[:500] + ("..." if len(text) > 500 else ""))

        # 텍스트 길이 분석
        lengths = [len(text) for text in texts]
        avg_len = np.mean(lengths)
        median_len = np.median(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)

        print(f"\n[텍스트 길이 통계]")
        print(f"  평균: {avg_len:.1f}자")
        print(f"  중앙값: {median_len:.1f}자")
        print(f"  최소: {min_len}자")
        print(f"  최대: {max_len}자")

        # 개체 반복 패턴 분석
        analyze_sample = min(len(texts), 300)
        print(f"\n[개체 반복 패턴 분석] (샘플 {analyze_sample}개)")
        repeated_count = 0
        all_repeated_entities = []
        pronoun_counts = []

        for text in tqdm(texts[:analyze_sample], desc="분석 중"):
            has_repeat, entities = analyze_nnp_repetition(text, kiwi)
            if has_repeat:
                repeated_count += 1
                all_repeated_entities.extend(entities)

            pronoun_count = count_pronouns(text, kiwi)
            pronoun_counts.append(pronoun_count)

        repeat_ratio = (repeated_count / analyze_sample) * 100
        avg_pronouns = np.mean(pronoun_counts) if pronoun_counts else 0

        print(f"  개체 반복 있는 샘플 비율: {repeat_ratio:.1f}%")
        print(f"  평균 대명사 개수: {avg_pronouns:.2f}개/샘플")

        if all_repeated_entities:
            entity_counter = Counter(all_repeated_entities)
            print(f"  가장 많이 반복된 개체 (Top 5):")
            for entity, count in entity_counter.most_common(5):
                print(f"    - {entity}: {count}회")

        # 적합성 평가
        print(f"\n[적합성 평가]")

        # 점수 계산
        length_score = min(5, max(1, avg_len / 100))  # 길이 기반 점수
        repeat_score = min(5, max(1, repeat_ratio / 10))  # 반복 비율 기반 점수
        pronoun_score = min(5, max(1, avg_pronouns * 2))  # 대명사 기반 점수

        total_score = (length_score + repeat_score + pronoun_score) / 3
        stars = round(total_score)

        print(f"  고유명사 반복 비율: {repeat_ratio:.1f}% ({'높음' if repeat_ratio > 30 else '보통' if repeat_ratio > 15 else '낮음'})")
        print(f"  평균 텍스트 길이: {avg_len:.0f}자 ({'적합' if avg_len > 200 else '짧음' if avg_len > 100 else '매우 짧음'})")
        print(f"  대명사 사용: {avg_pronouns:.2f}개 ({'많음' if avg_pronouns > 2 else '보통' if avg_pronouns > 0.5 else '적음'})")
        print(f"  종합 점수: {'⭐' * stars} ({stars}/5)")

        # 추천 사용량
        if isinstance(total_samples, str):
            if stars >= 4:
                recommended = 50000
            elif stars >= 3:
                recommended = 20000
            elif stars >= 2:
                recommended = 10000
            else:
                recommended = 5000
        else:
            if stars >= 4:
                recommended = min(total_samples, 50000)
            elif stars >= 3:
                recommended = min(total_samples, 20000)
            elif stars >= 2:
                recommended = min(total_samples, 10000)
            else:
                recommended = min(total_samples, 5000)

        print(f"  추천 사용량: {recommended/1000:.0f}K 샘플")

        return {
            'name': dataset_name + (f"/{subset}" if subset else ""),
            'total_samples': total_samples,
            'avg_length': avg_len,
            'median_length': median_len,
            'repeat_ratio': repeat_ratio,
            'avg_pronouns': avg_pronouns,
            'score': stars,
            'recommended': recommended,
            'text_field': text_field
        }

    except Exception as e:
        print(f"✗ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 8개 데이터셋 정의
datasets_to_analyze = [
    {
        'name': 'skt/kobest_v1',
        'subset': 'hellaswag',
        'split': 'train',
        'text_field': 'context',
        'streaming': False
    },
    {
        'name': 'HPLT/HPLT2.0_cleaned',
        'subset': 'kor_Hang',
        'split': 'train',
        'text_field': 'text',
        'streaming': True  # 너무 큰 데이터셋
    },
    {
        'name': 'nmixx-fin/ko-trans-us_news_retrieval',
        'subset': None,
        'split': 'train',
        'text_field': 'trans_text',
        'streaming': False
    },
    {
        'name': 'nmixx-fin/twice_kr_finance_news_summ',
        'subset': None,
        'split': 'train',
        'text_field': 'text',
        'streaming': False
    },
    {
        'name': 'nmixx-fin/twice_kr_news_bqa_cls',
        'subset': None,
        'split': 'train',
        'text_field': 'text',
        'streaming': False
    },
    {
        'name': 'nmixx-fin/twice_kr_fin_news_sent_cls',
        'subset': None,
        'split': 'train',
        'text_field': 'text',
        'streaming': False
    },
    {
        'name': 'dev7halo/naver-news-summarization-ko-with-gen',
        'subset': None,
        'split': 'train',
        'text_field': 'document',
        'streaming': False
    },
    {
        'name': 'AIR-Bench/qa_news_ko',
        'subset': None,
        'split': 'corpus_default',
        'text_field': 'text',
        'streaming': False
    }
]

# 분석 실행
results = []
for ds_config in datasets_to_analyze:
    result = analyze_dataset(
        dataset_name=ds_config['name'],
        subset=ds_config.get('subset'),
        split=ds_config['split'],
        text_field=ds_config['text_field'],
        sample_size=500,
        use_streaming=ds_config.get('streaming', False)
    )
    if result:
        results.append(result)
    print("\n")

# 최종 리포트
print("\n" + "="*80)
print("최종 분석 리포트")
print("="*80)

for i, result in enumerate(results, 1):
    total_str = f"{result['total_samples']:,}" if isinstance(result['total_samples'], int) else result['total_samples']
    print(f"\n데이터셋 {i}: {result['name']}")
    print(f"- 샘플 수: {total_str}")
    print(f"- 평균 길이: {result['avg_length']:.0f}자")
    print(f"- 개체 반복 비율: {result['repeat_ratio']:.1f}%")
    print(f"- 대명사 사용: {result['avg_pronouns']:.2f}개/샘플")
    print(f"- 적합성: {'⭐' * result['score']} ({result['score']}/5)")
    print(f"- 추천 사용량: {result['recommended']/1000:.0f}K 샘플")

# 우선순위 정렬
sorted_results = sorted(results, key=lambda x: (x['score'], x['repeat_ratio']), reverse=True)

print(f"\n\n{'='*80}")
print("최종 추천")
print("="*80)

high_priority = [r for r in sorted_results if r['score'] >= 4]
medium_priority = [r for r in sorted_results if 2 <= r['score'] < 4]
low_priority = [r for r in sorted_results if r['score'] < 2]

if high_priority:
    print("\n우선순위 1 (⭐⭐⭐⭐ 이상):")
    for r in high_priority:
        print(f"  - {r['name']}: {r['recommended']/1000:.0f}K 샘플 (반복률 {r['repeat_ratio']:.1f}%)")

if medium_priority:
    print("\n우선순위 2 (⭐⭐~⭐⭐⭐):")
    for r in medium_priority:
        print(f"  - {r['name']}: {r['recommended']/1000:.0f}K 샘플 (반복률 {r['repeat_ratio']:.1f}%)")

if low_priority:
    print("\n제외 권장 (⭐ 이하):")
    for r in low_priority:
        print(f"  - {r['name']}: 개체 반복 패턴 부족 (반복률 {r['repeat_ratio']:.1f}%)")

print(f"\n{'='*80}")
print("분석 완료!")
print("="*80)
