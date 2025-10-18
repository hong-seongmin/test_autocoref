#!/usr/bin/env python3
"""새 기준으로 테스트"""

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
        'pronoun_count': len(pronouns),
        'entity_count': len(entities),
        'pronoun_density': len(pronouns) / len(words) * 100,
        'pronoun_entity_ratio': len(pronouns) / max(1, len(entities)),
        'unique_pronouns': len(set(pronouns)),
    }

# 새 필터 기준
def new_filter(q):
    if not q:
        return False
    if q['pronoun_count'] < 2:
        return False
    if q['entity_count'] < 5:
        return False
    if q['pronoun_density'] < 1.0 or q['pronoun_density'] > 5.0:
        return False
    if q['pronoun_entity_ratio'] < 0.01 or q['pronoun_entity_ratio'] > 0.15:
        return False
    if q['unique_pronouns'] < 2:
        return False
    return True

print("Wikipedia 단락 테스트 (새 기준)")
print("="*80)

dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)

checked = 0
passed = 0
samples = []

for sample in dataset:
    if checked >= 500:
        break

    text = sample['text']
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 500 or len(para) > 2000:
            continue
        if para.startswith(('==', '##', '*', '-', '|')):
            continue
        if para.count('.') < 2:
            continue

        checked += 1
        q = analyze(para)

        if new_filter(q):
            passed += 1
            if len(samples) < 5:
                samples.append((para, q))

        if checked % 100 == 0:
            print(f"  체크: {checked}, 통과: {passed} ({passed/checked*100:.1f}%)")

print(f"\n최종: {checked}개 단락 중 {passed}개 통과 ({passed/checked*100:.1f}%)")

if samples:
    print("\n통과한 샘플 예시:")
    for i, (text, q) in enumerate(samples[:3]):
        print(f"\n[샘플 {i+1}]")
        print(f"  대명사: {q['pronoun_count']}, Entity: {q['entity_count']}")
        print(f"  밀도: {q['pronoun_density']:.2f}%, 비율: {q['pronoun_entity_ratio']:.3f}")
        print(f"  텍스트: {text[:150]}...")
