#!/usr/bin/env python3
"""
Test script for Long Sequence DeBERTa AutoML
"""

import sys
import os

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coref_automl.long_sequence_automl import run_long_sequence_automl

if __name__ == "__main__":
    print("Testing Long Sequence DeBERTa AutoML...")

    # 작은 규모로 테스트
    results = run_long_sequence_automl(
        model_name="kakaobank/kf-deberta-base",
        seq_lengths=[512, 768],  # 작은 길이로 테스트
        trials_per_length=2,      # 적은 trial로 테스트
        train_limit=5000          # 작은 데이터로 테스트
    )

    print("\nTest completed successfully!")
    print("Results:", results)