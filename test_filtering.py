#!/usr/bin/env python3
"""
필터링 기준 테스트 및 조정
"""

from datasets import load_dataset
from kiwipiepy import Kiwi
from typing import Dict, Any

kiwi = Kiwi()

def analyze_coref_quality(text: str) -> Dict[str, Any]:
    """품질 분석"""
    if not text or len(text) < 50:
        return None

    text = text[:1500]

    try:
        tokens = kiwi.tokenize(text)
    except:
        return None

    pronouns = []
    entities = []
    meaningful_words = []

    for token in tokens:
        if token.tag == 'NP':  # 대명사
            pronouns.append(token.form)
        elif token.tag in ['NNG', 'NNP', 'NNB']:  # 명사류
            if len(token.form) > 1:
                entities.append(token.form)

        if token.tag.startswith(('N', 'V', 'M', 'VA', 'VV')):
            meaningful_words.append(token.form)

    total_words = len(meaningful_words)
    if total_words == 0:
        return None

    pronoun_entity_ratio = len(pronouns) / max(1, len(entities))

    return {
        'text': text,
        'char_len': len(text),
        'pronoun_count': len(pronouns),
        'entity_count': len(entities),
        'total_words': total_words,
        'pronoun_density': len(pronouns) / max(1, total_words) * 100,  # 퍼센트
        'entity_density': len(entities) / max(1, total_words) * 100,
        'unique_pronouns': len(set(pronouns)),
        'unique_entities': len(set(entities)),
        'pronoun_entity_ratio': pronoun_entity_ratio,
    }


print("=" * 80)
print("Wikipedia 단락 샘플 분석")
print("=" * 80)

dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)

checked = 0
paragraphs_checked = 0
samples = []

for sample in dataset:
    if checked >= 100:  # 100개 문서만 체크
        break

    text = sample['text']
    checked += 1

    # \n\n로 단락 분리
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        paragraphs_checked += 1

        # 기본 필터
        if len(para) < 500 or len(para) > 2000:
            continue

        if para.startswith(('==', '##', '*', '-', '|')):
            continue

        if para.count('.') < 2:
            continue

        # Kiwi 분석
        quality = analyze_coref_quality(para)
        if quality:
            samples.append(quality)

print(f"\n체크: 문서 {checked}개, 단락 {paragraphs_checked}개")
print(f"품질 분석 완료: {len(samples)}개 단락\n")

# 통계 출력
if samples:
    print("=" * 80)
    print("품질 통계 (기본 필터 통과한 단락)")
    print("=" * 80)

    import numpy as np

    pronoun_counts = [s['pronoun_count'] for s in samples]
    entity_counts = [s['entity_count'] for s in samples]
    pronoun_densities = [s['pronoun_density'] for s in samples]
    pronoun_entity_ratios = [s['pronoun_entity_ratio'] for s in samples]

    print(f"\n대명사 개수:")
    print(f"  평균: {np.mean(pronoun_counts):.1f}")
    print(f"  범위: {np.min(pronoun_counts):.0f} - {np.max(pronoun_counts):.0f}")
    print(f"  중앙값: {np.median(pronoun_counts):.1f}")

    print(f"\nEntity 개수:")
    print(f"  평균: {np.mean(entity_counts):.1f}")
    print(f"  범위: {np.min(entity_counts):.0f} - {np.max(entity_counts):.0f}")
    print(f"  중앙값: {np.median(entity_counts):.1f}")

    print(f"\n대명사 밀도 (%):")
    print(f"  평균: {np.mean(pronoun_densities):.2f}%")
    print(f"  범위: {np.min(pronoun_densities):.2f}% - {np.max(pronoun_densities):.2f}%")
    print(f"  중앙값: {np.median(pronoun_densities):.2f}%")

    print(f"\nPronoun:Entity 비율:")
    print(f"  평균: {np.mean(pronoun_entity_ratios):.3f}")
    print(f"  범위: {np.min(pronoun_entity_ratios):.3f} - {np.max(pronoun_entity_ratios):.3f}")
    print(f"  중앙값: {np.median(pronoun_entity_ratios):.3f}")

    # 현재 필터 기준 통과율 체크
    print("\n" + "=" * 80)
    print("현재 필터 기준 통과율 체크")
    print("=" * 80)

    # 기준 1: 대명사 >= 3, entity >= 10
    pass1 = [s for s in samples if s['pronoun_count'] >= 3 and s['entity_count'] >= 10]
    print(f"\n기준 1 (대명사>=3, entity>=10): {len(pass1)}/{len(samples)} ({len(pass1)/len(samples)*100:.1f}%)")

    # 기준 2: 대명사 밀도 1.5-4.0%
    pass2 = [s for s in pass1 if 1.5 <= s['pronoun_density'] <= 4.0]
    print(f"기준 2 (밀도 1.5-4.0%): {len(pass2)}/{len(pass1)} ({len(pass2)/len(pass1)*100:.1f}% of pass1)")

    # 기준 3: Pronoun:Entity 비율 0.15-0.5
    pass3 = [s for s in pass2 if 0.15 <= s['pronoun_entity_ratio'] <= 0.5]
    print(f"기준 3 (비율 0.15-0.5): {len(pass3)}/{len(pass2)} ({len(pass3)/len(pass2)*100:.1f}% of pass2)")

    # 기준 4: unique_entities >= 5
    pass4 = [s for s in pass3 if s['unique_entities'] >= 5]
    print(f"기준 4 (unique_entities>=5): {len(pass4)}/{len(pass3)} ({len(pass4)/len(pass3)*100:.1f}% of pass3)")

    print(f"\n최종 통과율: {len(pass4)}/{len(samples)} ({len(pass4)/len(samples)*100:.1f}%)")

    # 각 단계별 실패 이유 분석
    print("\n" + "=" * 80)
    print("필터 실패 이유 분석")
    print("=" * 80)

    failed_at_1 = [s for s in samples if not (s['pronoun_count'] >= 3 and s['entity_count'] >= 10)]
    if failed_at_1:
        low_pronoun = [s for s in failed_at_1 if s['pronoun_count'] < 3]
        low_entity = [s for s in failed_at_1 if s['entity_count'] < 10]
        print(f"\n기준 1 실패 ({len(failed_at_1)}개):")
        print(f"  대명사 < 3: {len(low_pronoun)}개")
        print(f"  Entity < 10: {len(low_entity)}개")

    if pass1:
        failed_at_2 = [s for s in pass1 if not (1.5 <= s['pronoun_density'] <= 4.0)]
        if failed_at_2:
            too_low = [s for s in failed_at_2 if s['pronoun_density'] < 1.5]
            too_high = [s for s in failed_at_2 if s['pronoun_density'] > 4.0]
            print(f"\n기준 2 실패 ({len(failed_at_2)}개):")
            print(f"  밀도 < 1.5%: {len(too_low)}개 (평균: {np.mean([s['pronoun_density'] for s in too_low]):.2f}%)")
            print(f"  밀도 > 4.0%: {len(too_high)}개")

    # 샘플 텍스트 출력
    print("\n" + "=" * 80)
    print("통과한 샘플 예시")
    print("=" * 80)

    if pass4:
        for i, s in enumerate(pass4[:3]):
            print(f"\n[샘플 {i+1}]")
            print(f"  길이: {s['char_len']}자")
            print(f"  대명사: {s['pronoun_count']}개, Entity: {s['entity_count']}개")
            print(f"  밀도: {s['pronoun_density']:.2f}%, 비율: {s['pronoun_entity_ratio']:.3f}")
            preview = s['text'][:200] + "..." if len(s['text']) > 200 else s['text']
            print(f"  텍스트: {preview}")

    print("\n" + "=" * 80)
    print("실패한 샘플 예시 (기준 1 실패)")
    print("=" * 80)

    if failed_at_1:
        for i, s in enumerate(failed_at_1[:3]):
            print(f"\n[샘플 {i+1}]")
            print(f"  길이: {s['char_len']}자")
            print(f"  대명사: {s['pronoun_count']}개, Entity: {s['entity_count']}개")
            print(f"  밀도: {s['pronoun_density']:.2f}%, 비율: {s['pronoun_entity_ratio']:.3f}")
            preview = s['text'][:200] + "..." if len(s['text']) > 200 else s['text']
            print(f"  텍스트: {preview}")

    # 권장 기준 제시
    print("\n" + "=" * 80)
    print("권장 필터 기준 (10% 통과율 목표)")
    print("=" * 80)

    # 10% 통과율을 목표로 기준 완화
    pronoun_10th = np.percentile([s['pronoun_count'] for s in samples], 90)
    entity_10th = np.percentile([s['entity_count'] for s in samples], 10)
    density_10th = np.percentile([s['pronoun_density'] for s in samples], 90)

    print(f"\n대명사 개수 >= {int(pronoun_10th * 0.5)} (현재: >=3)")
    print(f"Entity 개수 >= {int(entity_10th)} (현재: >=10)")
    print(f"대명사 밀도 >= {density_10th * 0.7:.2f}% (현재: >=1.5%)")
    print(f"Pronoun:Entity 비율: 0.1-0.6 (현재: 0.15-0.5)")

else:
    print("⚠️ 분석 가능한 샘플이 없습니다!")
