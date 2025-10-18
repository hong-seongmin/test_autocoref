"""
결합된 데이터셋 품질 평가 스크립트
- combined_coref_1536, combined_coref_2048 분석
"""

from datasets import load_from_disk
from transformers import AutoTokenizer
from kiwipiepy import Kiwi
import numpy as np
import random

def analyze_text_quality(text: str, kiwi: Kiwi) -> dict:
    """텍스트 품질 분석"""
    if not text or len(text) < 50:
        return None

    # Kiwi 분석
    text_sample = text[:1500]
    try:
        tokens = kiwi.tokenize(text_sample)
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

    return {
        'text_len': len(text),
        'pronoun_count': len(pronouns),
        'entity_count': len(entities),
        'verb_count': len(verbs),
        'total_words': total_words,
        'pronoun_density': len(pronouns) / total_words * 100,
        'entity_density': len(entities) / total_words * 100,
        'unique_pronouns': len(set(pronouns)),
        'unique_entities': len(set(entities)),
        'pronoun_entity_ratio': len(pronouns) / max(1, len(entities)),
        'pronouns': pronouns,
        'entities': entities,
    }


def decode_and_analyze(dataset, tokenizer, kiwi, seq_len: int, sample_size: int = 200):
    """토큰화된 데이터셋을 디코딩하여 분석"""
    print(f"\n{'='*80}")
    print(f"📊 combined_coref_{seq_len} 품질 분석 (샘플: {sample_size}개)")
    print(f"{'='*80}")

    # 랜덤 샘플링
    total_samples = len(dataset)
    if sample_size > total_samples:
        sample_size = total_samples

    indices = random.sample(range(total_samples), sample_size)

    print(f"✅ 총 샘플 수: {total_samples:,}개")
    print(f"🔍 분석 샘플 수: {sample_size}개")
    print(f"\n분석 중...")

    # 텍스트 디코딩
    qualities = []
    text_lengths = []
    decoded_texts = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]

        # 디코딩
        input_ids = sample['input_ids']
        # PAD 토큰 제거
        non_pad_ids = [tid for tid in input_ids if tid != tokenizer.pad_token_id]
        decoded = tokenizer.decode(non_pad_ids, skip_special_tokens=True)

        text_lengths.append(len(decoded))
        decoded_texts.append(decoded)

        # 품질 분석
        quality = analyze_text_quality(decoded, kiwi)
        if quality:
            qualities.append(quality)

        if (i + 1) % 50 == 0:
            print(f"  진행: {i+1}/{sample_size}")

    # 통계 계산
    print(f"\n✅ 분석 완료: {len(qualities)}개 유효 샘플")

    if not qualities:
        print("⚠️ 유효한 샘플이 없습니다!")
        return None

    # 기본 통계
    print(f"\n{'='*80}")
    print("📈 기본 통계")
    print(f"{'='*80}")
    print(f"텍스트 길이 (문자 수):")
    print(f"  평균: {np.mean(text_lengths):.0f}자")
    print(f"  중앙값: {np.median(text_lengths):.0f}자")
    print(f"  범위: {np.min(text_lengths):.0f} ~ {np.max(text_lengths):.0f}자")

    # 대명사 통계
    pronoun_counts = [q['pronoun_count'] for q in qualities]
    pronoun_densities = [q['pronoun_density'] for q in qualities]

    print(f"\n대명사 개수:")
    print(f"  평균: {np.mean(pronoun_counts):.1f}개")
    print(f"  중앙값: {np.median(pronoun_counts):.0f}개")
    print(f"  범위: {np.min(pronoun_counts):.0f} ~ {np.max(pronoun_counts):.0f}개")

    print(f"\n대명사 밀도:")
    print(f"  평균: {np.mean(pronoun_densities):.2f}%")
    print(f"  중앙값: {np.median(pronoun_densities):.2f}%")
    print(f"  범위: {np.min(pronoun_densities):.2f}% ~ {np.max(pronoun_densities):.2f}%")
    print(f"  목표 범위(1.5-3.0%) 내: {sum(1.5 <= d <= 3.0 for d in pronoun_densities)}/{len(pronoun_densities)} ({sum(1.5 <= d <= 3.0 for d in pronoun_densities)/len(pronoun_densities)*100:.1f}%)")

    # Entity 통계
    entity_counts = [q['entity_count'] for q in qualities]
    entity_densities = [q['entity_density'] for q in qualities]

    print(f"\nEntity 개수:")
    print(f"  평균: {np.mean(entity_counts):.1f}개")
    print(f"  중앙값: {np.median(entity_counts):.0f}개")
    print(f"  범위: {np.min(entity_counts):.0f} ~ {np.max(entity_counts):.0f}개")

    # Pronoun:Entity 비율
    ratios = [q['pronoun_entity_ratio'] for q in qualities]
    print(f"\nPronoun:Entity 비율:")
    print(f"  평균: {np.mean(ratios):.3f} ({np.mean(ratios)*100:.1f}%)")
    print(f"  중앙값: {np.median(ratios):.3f} ({np.median(ratios)*100:.1f}%)")
    print(f"  범위: {np.min(ratios):.3f} ~ {np.max(ratios):.3f}")
    print(f"  목표 범위(0.01-0.15) 내: {sum(0.01 <= r <= 0.15 for r in ratios)}/{len(ratios)} ({sum(0.01 <= r <= 0.15 for r in ratios)/len(ratios)*100:.1f}%)")

    # Unique 통계
    unique_pronouns = [q['unique_pronouns'] for q in qualities]
    unique_entities = [q['unique_entities'] for q in qualities]

    print(f"\nUnique 대명사:")
    print(f"  평균: {np.mean(unique_pronouns):.1f}개")
    print(f"  중앙값: {np.median(unique_pronouns):.0f}개")

    print(f"\nUnique Entity:")
    print(f"  평균: {np.mean(unique_entities):.1f}개")
    print(f"  중앙값: {np.median(unique_entities):.0f}개")

    # 샘플 출력
    print(f"\n{'='*80}")
    print("📝 샘플 텍스트 예시 (3개)")
    print(f"{'='*80}")

    for i in range(min(3, len(decoded_texts))):
        text = decoded_texts[i]
        quality = qualities[i] if i < len(qualities) else None

        print(f"\n[샘플 {i+1}]")
        print(f"길이: {len(text)}자")
        if quality:
            print(f"대명사: {quality['pronoun_count']}개 (밀도: {quality['pronoun_density']:.2f}%)")
            print(f"Entity: {quality['entity_count']}개")
            print(f"비율: {quality['pronoun_entity_ratio']:.3f}")
            print(f"대명사 예시: {', '.join(quality['pronouns'][:10])}")
        print(f"\n텍스트 (앞 300자):")
        print(f"{text[:300]}...")

    return qualities


def main():
    print("=" * 80)
    print("🔍 결합된 데이터셋 품질 평가")
    print("=" * 80)

    # 로드
    print("\n📥 데이터셋 로딩...")
    ds_1536 = load_from_disk('prepared_datasets/combined_coref_1536')
    ds_2048 = load_from_disk('prepared_datasets/combined_coref_2048')

    print("\n📥 토크나이저 로딩...")
    tokenizer = AutoTokenizer.from_pretrained('kakaobank/kf-deberta-base')

    print("\n📥 Kiwi 초기화...")
    kiwi = Kiwi()

    # 기본 정보
    print(f"\n{'='*80}")
    print("📊 데이터셋 기본 정보")
    print(f"{'='*80}")
    print(f"combined_coref_1536: {len(ds_1536):,}개 샘플 (파일 크기: 446MB)")
    print(f"combined_coref_2048: {len(ds_2048):,}개 샘플 (파일 크기: 616MB)")
    print(f"총 샘플 수: {len(ds_1536) + len(ds_2048):,}개")

    # 분석 (각각 200개 샘플)
    qualities_1536 = decode_and_analyze(ds_1536, tokenizer, kiwi, 1536, sample_size=200)
    qualities_2048 = decode_and_analyze(ds_2048, tokenizer, kiwi, 2048, sample_size=200)

    # 종합 평가
    print(f"\n{'='*80}")
    print("🎯 종합 평가")
    print(f"{'='*80}")

    if qualities_1536 and qualities_2048:
        all_densities = [q['pronoun_density'] for q in qualities_1536 + qualities_2048]
        all_ratios = [q['pronoun_entity_ratio'] for q in qualities_1536 + qualities_2048]

        print(f"\n전체 평균:")
        print(f"  대명사 밀도: {np.mean(all_densities):.2f}% (목표: 1.5-3.0%)")
        print(f"  Pronoun:Entity 비율: {np.mean(all_ratios):.3f} ({np.mean(all_ratios)*100:.1f}%) (목표: 0.01-0.15)")

        # 품질 판정
        density_ok = 1.5 <= np.mean(all_densities) <= 3.5
        ratio_ok = 0.01 <= np.mean(all_ratios) <= 0.15

        print(f"\n품질 평가:")
        status_density = "✅ 양호" if density_ok else "⚠️ 범위 벗어남"
        status_ratio = "✅ 양호" if ratio_ok else "⚠️ 범위 벗어남"
        print(f"  대명사 밀도: {status_density}")
        print(f"  Pronoun:Entity 비율: {status_ratio}")

        if density_ok and ratio_ok:
            print(f"\n✅ 전체 평가: 우수 - Coreference resolution 학습에 적합")
        elif density_ok or ratio_ok:
            print(f"\n⚠️ 전체 평가: 보통 - 일부 개선 필요")
        else:
            print(f"\n❌ 전체 평가: 개선 필요 - 필터링 기준 재조정 권장")

    print(f"\n{'='*80}")
    print("평가 완료!")
    print(f"{'='*80}")


if __name__ == "__main__":
    random.seed(42)
    main()
