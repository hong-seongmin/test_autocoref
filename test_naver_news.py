#!/usr/bin/env python3
"""Naver News 필터링 기준 테스트"""

from datasets import load_dataset
from kiwipiepy import Kiwi

kiwi = Kiwi()

def analyze(text):
    text = text[:1500]
    try:
        tokens = kiwi.tokenize(text)
    except:
        return None

    pronouns = []
    entities = []
    words = []

    for token in tokens:
        if token.tag == 'NP':
            pronouns.append(token.form)
        elif token.tag in ['NNG', 'NNP', 'NNB']:
            if len(token.form) > 1:
                entities.append(token.form)
        if token.tag.startswith(('N', 'V', 'M', 'VA', 'VV')):
            words.append(token.form)

    if not words:
        return None

    return {
        'text': text,
        'char_len': len(text),
        'pronoun_count': len(pronouns),
        'entity_count': len(entities),
        'pronoun_density': len(pronouns) / len(words) * 100,
        'pronoun_entity_ratio': len(pronouns) / max(1, len(entities)),
        'unique_pronouns': len(set(pronouns)),
    }

print("Naver News 품질 분석")
print("="*80)

dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")
print(f"총 샘플 수: {len(dataset)}\n")

# 1000개 샘플 테스트
samples = []
for i in range(min(1000, len(dataset))):
    text = dataset[i]['document']

    # 최소 길이 체크
    if len(text) < 800:
        continue

    q = analyze(text)
    if q:
        samples.append(q)

print(f"분석 완료: {len(samples)}개 샘플\n")

import numpy as np

if samples:
    pronoun_counts = [s['pronoun_count'] for s in samples]
    entity_counts = [s['entity_count'] for s in samples]
    pronoun_densities = [s['pronoun_density'] for s in samples]
    pronoun_entity_ratios = [s['pronoun_entity_ratio'] for s in samples]
    char_lens = [s['char_len'] for s in samples]

    print("기본 통계")
    print("-"*80)
    print(f"문서 길이: 평균 {np.mean(char_lens):.0f}자 (범위: {np.min(char_lens)}-{np.max(char_lens)})")
    print(f"대명사 개수: 평균 {np.mean(pronoun_counts):.1f} (범위: {np.min(pronoun_counts)}-{np.max(pronoun_counts)})")
    print(f"Entity 개수: 평균 {np.mean(entity_counts):.1f} (범위: {np.min(entity_counts)}-{np.max(entity_counts)})")
    print(f"대명사 밀도: 평균 {np.mean(pronoun_densities):.2f}% (범위: {np.min(pronoun_densities):.2f}-{np.max(pronoun_densities):.2f}%)")
    print(f"Pronoun:Entity 비율: 평균 {np.mean(pronoun_entity_ratios):.3f}")

    # 필터 기준 테스트
    print("\n필터 기준별 통과율")
    print("-"*80)

    # 기준 1: 대명사 >= 3
    filter1 = [s for s in samples if s['pronoun_count'] >= 3]
    print(f"대명사 ≥3: {len(filter1)}/{len(samples)} ({len(filter1)/len(samples)*100:.1f}%)")

    # 기준 2: 대명사 밀도 >= 1.0%
    filter2 = [s for s in filter1 if s['pronoun_density'] >= 1.0]
    print(f"밀도 ≥1.0%: {len(filter2)}/{len(filter1)} ({len(filter2)/len(filter1)*100:.1f}% of filter1)")

    # 기준 3: Pronoun:Entity 비율 0.01-0.15
    filter3 = [s for s in filter2 if 0.01 <= s['pronoun_entity_ratio'] <= 0.15]
    print(f"비율 0.01-0.15: {len(filter3)}/{len(filter2)} ({len(filter3)/len(filter2)*100:.1f}% of filter2)")

    # 기준 4: unique pronouns >= 2
    filter4 = [s for s in filter3 if s['unique_pronouns'] >= 2]
    print(f"Unique ≥2: {len(filter4)}/{len(filter3)} ({len(filter4)/len(filter3)*100:.1f}% of filter3)")

    print(f"\n최종 통과율: {len(filter4)}/{len(samples)} ({len(filter4)/len(samples)*100:.1f}%)")

    # 완화된 기준 테스트
    print("\n완화된 기준 테스트")
    print("-"*80)

    # 완화 1: 대명사 >= 2
    relaxed1 = [s for s in samples if s['pronoun_count'] >= 2]
    print(f"대명사 ≥2: {len(relaxed1)}/{len(samples)} ({len(relaxed1)/len(samples)*100:.1f}%)")

    # 완화 2: 밀도 >= 0.8%
    relaxed2 = [s for s in relaxed1 if s['pronoun_density'] >= 0.8]
    print(f"밀도 ≥0.8%: {len(relaxed2)}/{len(relaxed1)} ({len(relaxed2)/len(relaxed1)*100:.1f}%)")

    # 완화 3: 비율 0.005-0.2 (더 넓게)
    relaxed3 = [s for s in relaxed2 if 0.005 <= s['pronoun_entity_ratio'] <= 0.2]
    print(f"비율 0.005-0.2: {len(relaxed3)}/{len(relaxed2)} ({len(relaxed3)/len(relaxed2)*100:.1f}%)")

    print(f"\n완화 기준 통과율: {len(relaxed3)}/{len(samples)} ({len(relaxed3)/len(samples)*100:.1f}%)")

    # 통과한 샘플 예시
    if filter4:
        print("\n통과 샘플 예시 (엄격한 기준)")
        print("-"*80)
        for i, s in enumerate(filter4[:3]):
            print(f"\n[샘플 {i+1}]")
            print(f"  길이: {s['char_len']}자")
            print(f"  대명사: {s['pronoun_count']}, Entity: {s['entity_count']}, 밀도: {s['pronoun_density']:.2f}%")
            print(f"  텍스트: {s['text'][:200]}...")

    # 백분위수 분석
    print("\n백분위수 분석 (데이터 분포)")
    print("-"*80)
    for pct in [10, 25, 50, 75, 90]:
        p_count = np.percentile(pronoun_counts, pct)
        p_density = np.percentile(pronoun_densities, pct)
        print(f"{pct}%: 대명사 {p_count:.1f}개, 밀도 {p_density:.2f}%")

    print("\n권장 기준")
    print("-"*80)
    print("엄격 (5-10% 통과): 대명사 ≥3, 밀도 ≥1.0%, 비율 0.01-0.15")
    print("적절 (15-25% 통과): 대명사 ≥2, 밀도 ≥0.8%, 비율 0.005-0.2")
    print("완화 (30-40% 통과): 대명사 ≥2, 밀도 ≥0.6%, 비율 0.005-0.25")
