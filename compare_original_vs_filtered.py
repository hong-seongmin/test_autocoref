#!/usr/bin/env python3
"""
Compare original vs filtered dataset quality
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from kiwipiepy import Kiwi
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("kakaobank/kf-deberta-base")
kiwi = Kiwi()

KOREAN_PRONOUNS = ['그', '그녀', '그들', '이', '저', '이것', '그것', '저것', '자신', '본인', '당신']

def analyze_text(text):
    """Quick pronoun analysis"""
    if not text or len(text) < 100:
        return None

    try:
        result = kiwi.analyze(text)
        pronouns = []
        for token in result[0][0]:
            if token.tag in ['NP', 'NNG']:
                if any(p in token.form for p in KOREAN_PRONOUNS):
                    pronouns.append(token.form)

        pronoun_count = len(pronouns)
        pronoun_density = (pronoun_count / len(text) * 100) if len(text) > 0 else 0

        return {
            'char_len': len(text),
            'pronoun_count': pronoun_count,
            'pronoun_density': pronoun_density,
            'has_pronouns': pronoun_count >= 2
        }
    except:
        return None

print("="*80)
print("ORIGINAL KLUE MRC DATASET (first 1000 samples)")
print("="*80)

klue_original = load_dataset("klue", "mrc", split="train")
klue_results = []

for i in range(min(1000, len(klue_original))):
    text = klue_original[i]['context']
    result = analyze_text(text)
    if result:
        klue_results.append(result)

print(f"Samples analyzed: {len(klue_results)}")
print(f"Avg char length: {np.mean([r['char_len'] for r in klue_results]):.0f}")
print(f"Avg pronoun count: {np.mean([r['pronoun_count'] for r in klue_results]):.2f}")
print(f"Avg pronoun density: {np.mean([r['pronoun_density'] for r in klue_results]):.3f}%")
print(f"Samples with ≥2 pronouns: {sum(r['has_pronouns'] for r in klue_results)/len(klue_results)*100:.1f}%")

print("\n" + "="*80)
print("ORIGINAL WIKIPEDIA DATASET (first 500 samples)")
print("="*80)

wiki_original = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
wiki_results = []
count = 0

for item in wiki_original:
    if count >= 500:
        break

    text = item['text']
    result = analyze_text(text)
    if result:
        wiki_results.append(result)
    count += 1

print(f"Samples analyzed: {len(wiki_results)}")
print(f"Avg char length: {np.mean([r['char_len'] for r in wiki_results]):.0f}")
print(f"Avg pronoun count: {np.mean([r['pronoun_count'] for r in wiki_results]):.2f}")
print(f"Avg pronoun density: {np.mean([r['pronoun_density'] for r in wiki_results]):.3f}%")
print(f"Samples with ≥2 pronouns: {sum(r['has_pronouns'] for r in wiki_results)/len(wiki_results)*100:.1f}%")

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print("\nKLUE MRC:")
print(f"  Original: {np.mean([r['pronoun_density'] for r in klue_results]):.3f}% density, {sum(r['has_pronouns'] for r in klue_results)/len(klue_results)*100:.1f}% coverage")
print(f"  Filtered: 1.128% density, 98.0% coverage")
print(f"  Improvement: {(1.128 - np.mean([r['pronoun_density'] for r in klue_results])) / np.mean([r['pronoun_density'] for r in klue_results]) * 100:+.1f}% density")
print(f"  Sample retention: 1362/{len(klue_original)} = {1362/len(klue_original)*100:.1f}%")

print("\nWikipedia:")
print(f"  Original: {np.mean([r['pronoun_density'] for r in wiki_results]):.3f}% density, {sum(r['has_pronouns'] for r in wiki_results)/len(wiki_results)*100:.1f}% coverage")
print(f"  Filtered: 1.264% density, 100.0% coverage")
print(f"  Improvement: {(1.264 - np.mean([r['pronoun_density'] for r in wiki_results])) / np.mean([r['pronoun_density'] for r in wiki_results]) * 100:+.1f}% density")
print(f"  Note: Unknown total sample count (streaming dataset with early termination)")
