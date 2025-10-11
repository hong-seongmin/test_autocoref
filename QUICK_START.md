# 빠른 시작 가이드

## 1️⃣ 데이터셋 준비 (한 번만 실행)

```bash
# 모든 데이터셋 생성 (30-40분 소요)
python scripts/prepare_filtered_datasets.py
```

**생성되는 데이터셋**:
- KLUE MRC (1536, 2048) - 각 ~1,200개
- Wikipedia (1536, 2048) - 각 ~50,000개
- Naver News (1536, 2048) - 각 ~3,000-5,000개

**총 ~108,000개 샘플, 평균 대명사 밀도 2.5%**

## 2️⃣ 학습 실행

```bash
# 간편 실행 (권장)
bash run_training.sh
```

또는 수동 실행:

```bash
python -m coref_automl.long_sequence_automl \
    --model kakaobank/kf-deberta-base \
    --seq-lengths 1536 2048 \
    --trials 30 \
    --epoch-choices 2 3 \
    --dataset-choice prepared_datasets/klue_mrc_filtered_1536 \
    --dataset-choice prepared_datasets/klue_mrc_filtered_2048 \
    --dataset-choice prepared_datasets/wikipedia_filtered_1536 \
    --dataset-choice prepared_datasets/wikipedia_filtered_2048 \
    --dataset-choice prepared_datasets/naver_news_filtered_1536 \
    --dataset-choice prepared_datasets/naver_news_filtered_2048
```

## 3️⃣ 빠른 테스트 (선택)

데이터셋 준비가 너무 오래 걸리면 소량으로 테스트:

```bash
# Wikipedia 5,000개만 (5분)
python scripts/prepare_filtered_datasets.py \
    --datasets wikipedia \
    --seq-lengths 2048 \
    --wiki-samples 5000

# 빠른 학습 (5 trials)
python -m coref_automl.long_sequence_automl \
    --model kakaobank/kf-deberta-base \
    --seq-lengths 2048 \
    --trials 5 \
    --dataset-choice prepared_datasets/wikipedia_filtered_2048
```

## 📊 예상 결과

- **Top5 점수**: 60.44% (기존) → **65-70% (목표)**
- **데이터 품질**: 대명사 밀도 1.2% → **2.5%** (+108%)
- **데이터 양**: 7,500개 → **108,000개** (+1,340%)

## 🔧 주요 옵션

### 데이터셋 준비

```bash
# 특정 데이터셋만
--datasets klue_mrc wikipedia naver_news

# 시퀀스 길이 선택
--seq-lengths 1536 2048

# Wikipedia 샘플 수 조정
--wiki-samples 50000  # 기본값
```

### 학습

```bash
# Trial 수 (하이퍼파라미터 탐색)
--trials 30  # 많을수록 좋지만 느림

# 에폭 선택지
--epoch-choices 2 3  # Optuna가 자동 선택

# 시퀀스 길이
--seq-lengths 1536 2048  # 여러 개 가능
```

## 📝 참고

- 전체 문서: `DATASET_README.md`
- 품질 분석: `python analyze_filtered_datasets.py`
- 진행 확인: `tail -f /tmp/full_dataset_prep.log`
