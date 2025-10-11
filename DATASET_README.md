# 개선된 Coreference Resolution 데이터셋 준비

## 📋 개요

이 프로젝트는 한국어 상호참조 해결(Coreference Resolution)을 위한 MLM(Masked Language Model) 학습 데이터를 준비합니다.

### 핵심 개선사항

1. **Wikipedia 단락 분리**: `\n\n`로 단락 분리하여 적절한 길이(500-2000자)로 처리
2. **실제 데이터 기반 필터링**: 이론과 실제 데이터 분석을 통한 최적 기준 적용
3. **3개 데이터셋 통합**: KLUE MRC + Wikipedia + Naver News
4. **다중 시퀀스 길이**: 1536, 2048 토큰 모두 지원
5. **고속 처리**: 멀티프로세싱으로 50-100배 속도 향상

## 🎯 필터링 기준

### 공통 기준
- **대명사 개수**: ≥2개
- **Entity 개수**: ≥5개
- **대명사 밀도**: 1.0-5.0%
- **Pronoun:Entity 비율**: 0.01-0.15 (1-15%)
- **Unique pronouns**: ≥2개

### 데이터셋별 추가 기준

| 데이터셋 | 추가 기준 | 예상 통과율 | 목표 샘플 수 |
|---------|----------|------------|------------|
| **KLUE MRC** | Entity ≥8, Unique pronouns ≥2 | ~7% | ~1,200 |
| **Wikipedia** | Unique pronouns ≥2 | ~22% | ~50,000 |
| **Naver News** | 밀도 ≥0.8%, 비율 ≥0.005 | ~15-25% | ~3,000-5,000 |

## 🚀 사용 방법

### 1단계: 데이터셋 준비

#### 기본 사용 (모든 데이터셋, 두 시퀀스 길이)

```bash
python scripts/prepare_filtered_datasets.py
```

이 명령어는:
- KLUE MRC, Wikipedia, Naver News 모두 생성
- 1536, 2048 시퀀스 길이 모두 생성
- Wikipedia 50,000개 단락 수집
- 20개 워커로 병렬 처리

#### 커스텀 옵션

```bash
# Wikipedia만, 더 많은 샘플
python scripts/prepare_filtered_datasets.py \
    --datasets wikipedia \
    --seq-lengths 2048 \
    --wiki-samples 100000

# 특정 데이터셋만
python scripts/prepare_filtered_datasets.py \
    --datasets klue_mrc wikipedia \
    --seq-lengths 1536 2048

# 워커 수 조정
python scripts/prepare_filtered_datasets.py \
    --num-workers 10
```

### 2단계: 학습 실행

#### 방법 1: 간편 스크립트 사용 (권장)

```bash
bash run_training.sh
```

이 스크립트는:
- 생성된 모든 데이터셋 자동 감지
- 최적 하이퍼파라미터로 Optuna 실행
- 30개 trial로 최고 성능 탐색

#### 방법 2: 수동 실행

```bash
# 모든 데이터셋 사용
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

```bash
# 특정 데이터셋만 (예: Wikipedia 2048만)
python -m coref_automl.long_sequence_automl \
    --model kakaobank/kf-deberta-base \
    --seq-lengths 2048 \
    --trials 20 \
    --dataset-choice prepared_datasets/wikipedia_filtered_2048
```

## 📊 예상 결과

### 데이터셋 크기

| 데이터셋 | 시퀀스 1536 | 시퀀스 2048 | 평균 대명사 밀도 |
|---------|------------|------------|-----------------|
| KLUE MRC | ~1,200개 | ~1,200개 | 2.4% |
| Wikipedia | ~50,000개 | ~50,000개 | 2.6% |
| Naver News | ~3,000-5,000개 | ~3,000-5,000개 | 1.5-2.0% |
| **합계** | **~54,000개** | **~54,000개** | **~2.5%** |

### 처리 시간

- KLUE MRC: ~30초
- Wikipedia 50,000개: ~20-30분
- Naver News: ~2-5분
- **총 처리 시간: 약 30-40분**

### 성능 향상 예상

| 항목 | 기존 | 개선 후 | 향상 |
|------|------|--------|------|
| 데이터셋 크기 | 7,500개 | 54,000개 | **+620%** |
| 평균 대명사 밀도 | 1.2% | 2.5% | **+108%** |
| Top5 점수 (목표) | 60.44% | 65-70% | **+7-16%** |

## 🔍 품질 검증

### 생성된 데이터셋 분석

```bash
python analyze_filtered_datasets.py
```

이 스크립트는:
- 각 데이터셋의 품질 통계 출력
- 샘플 텍스트 예시 제공
- 대명사/Entity 분포 분석

### 개별 데이터셋 테스트

```bash
# Wikipedia 기준 테스트
python test_new_criteria.py

# Naver News 기준 테스트
python test_naver_news.py
```

## 📁 파일 구조

```
corefer/
├── scripts/
│   └── prepare_filtered_datasets.py  # 메인 데이터 준비 스크립트
├── prepared_datasets/                # 생성된 데이터셋 저장
│   ├── klue_mrc_filtered_1536/
│   ├── klue_mrc_filtered_2048/
│   ├── wikipedia_filtered_1536/
│   ├── wikipedia_filtered_2048/
│   ├── naver_news_filtered_1536/
│   └── naver_news_filtered_2048/
├── run_training.sh                   # 간편 학습 실행 스크립트
├── analyze_filtered_datasets.py      # 품질 분석 도구
├── test_new_criteria.py              # Wikipedia 필터 테스트
├── test_naver_news.py                # Naver News 필터 테스트
└── DATASET_README.md                 # 이 문서
```

## 🛠️ 주요 인자 설명

### prepare_filtered_datasets.py

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | kakaobank/kf-deberta-base | 토크나이저 모델 |
| `--seq-lengths` | 1536 2048 | 시퀀스 길이 목록 |
| `--datasets` | all | 생성할 데이터셋 (klue_mrc, wikipedia, naver_news, all) |
| `--wiki-samples` | 50000 | Wikipedia 최대 단락 수 |
| `--num-workers` | CPU-4 | 병렬 처리 워커 수 |
| `--save-dir` | ./prepared_datasets | 저장 디렉토리 |

### long_sequence_automl.py

| 인자 | 설명 | 예시 |
|------|------|------|
| `--model` | 사용할 모델 | kakaobank/kf-deberta-base |
| `--seq-lengths` | 탐색할 시퀀스 길이 | 1536 2048 |
| `--trials` | Optuna trial 수 | 30 |
| `--epoch-choices` | 에폭 선택지 | 2 3 |
| `--dataset-choice` | 데이터셋 경로 (여러 개 가능) | prepared_datasets/... |

## 💡 팁

### 빠른 테스트

소량으로 빠르게 테스트하려면:

```bash
# Wikipedia 5,000개만
python scripts/prepare_filtered_datasets.py \
    --datasets wikipedia \
    --seq-lengths 2048 \
    --wiki-samples 5000

# 학습도 빠르게 (5 trials)
python -m coref_automl.long_sequence_automl \
    --trials 5 \
    --dataset-choice prepared_datasets/wikipedia_filtered_2048
```

### 메모리 부족 시

```bash
# 워커 수 줄이기
python scripts/prepare_filtered_datasets.py --num-workers 8

# 또는 데이터셋 하나씩 생성
python scripts/prepare_filtered_datasets.py --datasets klue_mrc
python scripts/prepare_filtered_datasets.py --datasets wikipedia
python scripts/prepare_filtered_datasets.py --datasets naver_news
```

### 진행 상황 모니터링

백그라운드 실행 시:

```bash
python scripts/prepare_filtered_datasets.py 2>&1 | tee dataset_prep.log &

# 다른 터미널에서
tail -f dataset_prep.log
```

## 📈 이론적 근거

### 왜 대명사 밀도 1.0-5.0%인가?

**SpanBERT 연구 (2019)**:
- Span masking으로 OntoNotes 79.6% F1 달성
- Entity mention 전체를 마스킹하여 학습
- 대명사와 명사가 적절히 혼합된 데이터가 효과적

**실제 자연스러운 텍스트**:
- 대명사 너무 많으면 (>5%): 문맥 모호성 증가
- 대명사 너무 적으면 (<1%): Coreference 학습 부족
- **최적: 1.5-3.0%** (우리 목표)

### 왜 Entity:Pronoun 비율이 중요한가?

**실제 데이터 분석 결과**:
- Wikipedia: 평균 100개 entity, 2-3개 대명사 → 비율 0.02-0.03 (2-3%)
- 비율 0.15 (15%) 요구 → 100% 실패
- **조정: 0.01-0.15 (1-15%)** → 22% 통과율 달성

### Wikipedia 단락 분리의 효과

**문제**: 긴 문서 (평균 2,628자) → 2048 토큰 초과 → 정보 손실

**해결**: `\n\n` 단락 분리
- 647,897개 문서 → 5,232,647개 단락
- 단락당 500-2000자 (적절한 길이)
- **결과: +76% 더 많은 고품질 샘플**

## 🔧 트러블슈팅

### Q: "0개 수집" 메시지가 계속 나옵니다

A: Wikipedia는 처음에 낮은 통과율을 보입니다. 인내심을 가지고 기다리세요. 약 3,000-10,000개 문서 후 샘플이 모이기 시작합니다.

### Q: Naver News가 너무 적게 수집됩니다

A: Naver News는 대명사 사용이 적은 편입니다. 15-25% 통과율이 정상입니다 (~3,000-5,000개).

### Q: 메모리 에러가 발생합니다

A: `--num-workers` 줄이거나 데이터셋을 하나씩 생성하세요.

### Q: 학습이 너무 느립니다

A:
- 더 적은 데이터로 테스트: `--wiki-samples 10000`
- 더 적은 trials: `--trials 10`
- 단일 시퀀스 길이: `--seq-lengths 2048`

## 📚 참고 문헌

- **SpanBERT** (Joshi et al., 2019): Span masking for coreference
- **Don't Stop Pretraining** (Gururangan et al., 2020): DAPT/TAPT
- **OntoNotes CoNLL-2012**: Coreference resolution benchmark
- **KLUE Benchmark** (Park et al., 2021): Korean NLU tasks

## 📞 문의

문제가 발생하면:
1. `analyze_filtered_datasets.py`로 품질 확인
2. `test_*.py` 스크립트로 필터 기준 검증
3. 로그 파일 확인
