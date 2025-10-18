#!/usr/bin/env python3
"""
Analyze the quality of filtered datasets
"""

import json
from datasets import load_from_disk
from transformers import AutoTokenizer
from kiwipiepy import Kiwi
import numpy as np
from collections import Counter

# Initialize tools
tokenizer = AutoTokenizer.from_pretrained("kakaobank/kf-deberta-base")
kiwi = Kiwi()

# Korean pronouns to check
KOREAN_PRONOUNS = ['그', '그녀', '그들', '이', '저', '이것', '그것', '저것', '자신', '본인', '당신']

def analyze_sample(sample_dict):
    """Analyze a single sample"""
    input_ids = sample_dict['input_ids']

    # Decode text
    text = tokenizer.decode(input_ids, skip_special_tokens=True)

    # Basic stats
    char_len = len(text)
    token_len = len(input_ids)

    # Kiwi analysis
    try:
        result = kiwi.analyze(text)

        # Count pronouns
        pronouns = []
        entities = []

        for token in result[0][0]:
            if token.tag in ['NP', 'NNG']:  # Pronouns and proper nouns
                if any(p in token.form for p in KOREAN_PRONOUNS):
                    pronouns.append(token.form)
                elif token.tag == 'NP':
                    entities.append(token.form)
            elif token.tag == 'NNP':  # Proper nouns (entities)
                entities.append(token.form)

        pronoun_count = len(pronouns)
        entity_count = len(entities)
        unique_pronouns = len(set(pronouns))

        # Calculate pronoun density
        pronoun_density = (pronoun_count / char_len * 100) if char_len > 0 else 0

    except Exception as e:
        pronoun_count = 0
        entity_count = 0
        unique_pronouns = 0
        pronoun_density = 0.0

    return {
        'text': text,
        'char_len': char_len,
        'token_len': token_len,
        'pronoun_count': pronoun_count,
        'entity_count': entity_count,
        'unique_pronouns': unique_pronouns,
        'pronoun_density': pronoun_density
    }

def analyze_dataset(dataset_path, dataset_name, num_samples=50):
    """Analyze a dataset"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"{'='*80}")

    # Load dataset
    dataset = load_from_disk(dataset_path)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")

    # Sample for analysis
    sample_size = min(num_samples, total_samples)
    indices = np.linspace(0, total_samples - 1, sample_size, dtype=int)

    results = []
    for idx in indices:
        sample = dataset[int(idx)]
        result = analyze_sample(sample)
        results.append(result)

    # Calculate statistics
    char_lens = [r['char_len'] for r in results]
    token_lens = [r['token_len'] for r in results]
    pronoun_counts = [r['pronoun_count'] for r in results]
    entity_counts = [r['entity_count'] for r in results]
    pronoun_densities = [r['pronoun_density'] for r in results]
    unique_pronouns = [r['unique_pronouns'] for r in results]

    print(f"\n--- Statistics (from {sample_size} samples) ---")
    print(f"Character length: avg={np.mean(char_lens):.0f}, min={np.min(char_lens)}, max={np.max(char_lens)}")
    print(f"Token length: avg={np.mean(token_lens):.0f}, min={np.min(token_lens)}, max={np.max(token_lens)}")
    print(f"Pronoun count: avg={np.mean(pronoun_counts):.2f}, min={np.min(pronoun_counts)}, max={np.max(pronoun_counts)}")
    print(f"Entity count: avg={np.mean(entity_counts):.2f}, min={np.min(entity_counts)}, max={np.max(entity_counts)}")
    print(f"Unique pronouns: avg={np.mean(unique_pronouns):.2f}, min={np.min(unique_pronouns)}, max={np.max(unique_pronouns)}")
    print(f"Pronoun density (%): avg={np.mean(pronoun_densities):.3f}, min={np.min(pronoun_densities):.3f}, max={np.max(pronoun_densities):.3f}")

    # Quality checks
    samples_with_pronouns = sum(1 for p in pronoun_counts if p >= 2)
    print(f"\nSamples with ≥2 pronouns: {samples_with_pronouns}/{sample_size} ({samples_with_pronouns/sample_size*100:.1f}%)")

    samples_with_entities = sum(1 for e in entity_counts if e >= 3)
    print(f"Samples with ≥3 entities: {samples_with_entities}/{sample_size} ({samples_with_entities/sample_size*100:.1f}%)")

    high_density_samples = sum(1 for d in pronoun_densities if d >= 1.5)
    print(f"Samples with pronoun density ≥1.5%: {high_density_samples}/{sample_size} ({high_density_samples/sample_size*100:.1f}%)")

    # Show 3 sample texts
    print(f"\n--- Sample Texts (first 3) ---")
    for i in range(min(3, len(results))):
        r = results[i]
        preview = r['text'][:300] + "..." if len(r['text']) > 300 else r['text']
        print(f"\n[Sample {i+1}]")
        print(f"  Length: {r['char_len']} chars, {r['token_len']} tokens")
        print(f"  Pronouns: {r['pronoun_count']}, Entities: {r['entity_count']}, Density: {r['pronoun_density']:.3f}%")
        print(f"  Text: {preview}")

    return {
        'total_samples': total_samples,
        'avg_char_len': np.mean(char_lens),
        'avg_pronoun_count': np.mean(pronoun_counts),
        'avg_entity_count': np.mean(entity_counts),
        'avg_pronoun_density': np.mean(pronoun_densities),
        'pct_with_pronouns': samples_with_pronouns/sample_size*100,
        'pct_with_entities': samples_with_entities/sample_size*100,
        'pct_high_density': high_density_samples/sample_size*100,
    }

if __name__ == "__main__":
    datasets = [
        ("prepared_datasets/klue_mrc_filtered_1536", "KLUE MRC (1536)"),
        ("prepared_datasets/klue_mrc_filtered_2048", "KLUE MRC (2048)"),
        ("prepared_datasets/wikipedia_filtered_1536", "Wikipedia (1536)"),
        ("prepared_datasets/wikipedia_filtered_2048", "Wikipedia (2048)"),
    ]

    all_results = {}

    for path, name in datasets:
        result = analyze_dataset(path, name, num_samples=50)
        all_results[name] = result

    # Summary comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Dataset':<25} {'Samples':<10} {'Avg Chars':<12} {'Avg Pron':<10} {'Avg Ent':<10} {'Density%':<10} {'Quality%':<10}")
    print("-" * 95)

    for name, result in all_results.items():
        print(f"{name:<25} {result['total_samples']:<10} {result['avg_char_len']:<12.0f} "
              f"{result['avg_pronoun_count']:<10.2f} {result['avg_entity_count']:<10.2f} "
              f"{result['avg_pronoun_density']:<10.3f} {result['pct_high_density']:<10.1f}")

    print("\nQuality% = Percentage of samples with pronoun density ≥1.5%")
