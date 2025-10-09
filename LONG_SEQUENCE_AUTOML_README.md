# Long Sequence DeBERTa AutoML

DeBERTa 모델의 긴 시퀀스 (1024-2048 tokens) coreference resolution 성능을 극대화하기 위한 자동화된 최적화 시스템입니다.

## 주요 특징

### 1. 긴 시퀀스 지원
- **지원 길이**: 512, 768, 1024, 1280, 1536, 1792, 2048 tokens
- **메모리 최적화**: 자동 배치 크기 및 gradient accumulation 조정
- **안정성**: 긴 시퀀스 특화 학습 기법 적용

### 2. 향상된 데이터셋
- **다중 소스 통합**: Wikipedia, KLUE 뉴스, KorQuAD 등
- **품질 기반 필터링**: 길이, 도메인, coreference 밀도 기반
- **동적 샘플링**: 품질 가중치 기반 스마트 샘플링

### 3. 고급 하이퍼파라미터 최적화
- **길이별 특화**: 시퀀스 길이에 따른 최적 파라미터 범위
- **메모리 인식**: GPU 메모리 제약 자동 고려
- **다차원 탐색**: LR, batch size, gradient accumulation 동시 최적화

## 설치 및 준비

```bash
# 의존성 설치
pip install -r requirements.txt

# Hugging Face 토큰 설정 (선택사항)
export HF_TOKEN="your_token_here"
```

## 사용법

### 기본 실행

```bash
# 기본 설정으로 실행 (1024, 1536, 2048 tokens)
python -m coref_automl.long_sequence_automl \
    --model kakaobank/kf-deberta-base \
    --trials 20
```

### 고급 설정

```bash
# 커스텀 설정으로 실행
python -m coref_automl.long_sequence_automl \
    --model kakaobank/kf-deberta-base \
    --seq-lengths 1024 1536 2048 \
    --trials 30 \
    --train-limit 50000
```

### 테스트 실행

```bash
# 작은 규모로 테스트
python test_long_sequence_automl.py
```

## 시스템 구성

### 데이터 저장 위치 변경
기존 `/tmp/coref_automl_bus.ndjson`에서 `data/coref_automl_bus.ndjson`으로 변경되었습니다.

### 메모리 기반 배치 최적화
시퀀스 길이에 따라 자동으로 배치 크기와 gradient accumulation을 조정합니다:

| 길이 | 배치 크기 | Grad Acc | 메모리 사용량 |
|------|-----------|----------|---------------|
| 512  | 32        | 1        | ~4GB         |
| 1024 | 16        | 2        | ~8GB         |
| 1536 | 8         | 4        | ~12GB        |
| 2048 | 4         | 8        | ~16GB        |

### 하이퍼파라미터 범위

#### 512 tokens
```python
lr: (1e-5, 5e-4)
warmup_ratio: (0.0, 0.2)
weight_decay: (0.0, 0.1)
grad_acc: [1, 2]
```

#### 1024 tokens
```python
lr: (5e-6, 2e-4)
warmup_ratio: (0.05, 0.25)
weight_decay: (0.01, 0.08)
grad_acc: [1, 2, 4]
```

#### 1536+ tokens
```python
lr: (1e-6, 1e-4)
warmup_ratio: (0.1, 0.3)
weight_decay: (0.02, 0.06)
grad_acc: [2, 4, 8]
```

## 모니터링 및 결과

### 실시간 모니터링
```bash
# 대시보드 실행
python -m coref_automl.dashboard
```

### 결과 분석
```bash
# 결과 파일 확인
cat data/coref_automl_bus.ndjson | jq 'select(.section == "eval_stream")' | tail -10

# 최고 성능 확인
grep "eval_stream" data/coref_automl_bus.ndjson | jq -r 'select(.coref_top5) | "\(.seq_len)t: \(.coref_top5)"' | sort -n -r | head -5
```

## 결과 해석

### 메트릭 설명
- **LAMBADA Top1**: 언어 이해 정확도
- **Coref F1**: Coreference resolution 품질 (precision + recall)
- **Coref Top5**: 상위 5개 예측 중 정답 포함 비율

### 성능 목표
- **현재**: Coref Top5 ~60%
- **목표**: Coref Top5 65-70%
- **향상**: 15-20% 성능 향상 예상

## 문제 해결

### 메모리 부족
```python
# 배치 크기 줄이기
batch_config = compute_optimal_batch_config(seq_len, model_name, available_memory_gb=40)
```

### 학습 불안정
```python
# 더 낮은 learning rate 사용
lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
```

### 데이터 로드 실패
```python
# 대안 데이터 소스 사용
# KLUE 데이터가 실패하면 Wikipedia만 사용
configs = [config for config in configs if config.source != "klue"]
```

## 확장 및 커스터마이징

### 새로운 데이터 소스 추가
```python
@dataclass
class DatasetConfig:
    source: str = "custom_dataset"
    subset: Optional[str] = None
    split: str = "train"
    domain: str = "custom"
    quality_weight: float = 0.8
    min_length: int = 512
    max_length: int = 2048

# preprocess_domain_data 함수에 custom 로직 추가
```

### 커스텀 평가 메트릭
```python
def custom_evaluation_metric(model, eval_data):
    # 커스텀 평가 로직
    pass
```

## 성능 최적화 팁

1. **메모리 관리**: GPU 메모리의 80% 이내로 사용
2. **학습 안정화**: 긴 시퀀스는 낮은 LR과 긴 warmup 사용
3. **데이터 품질**: Coreference가 풍부한 텍스트 우선
4. **평가 일관성**: 동일한 평가 데이터로 비교

## 다음 단계

1. **현재 실행 결과 분석**
2. **최적 파라미터 추출 및 적용**
3. **더 큰 데이터셋으로 확장**
4. **앙상블 모델링 고려**

이 시스템으로 DeBERTa의 긴 시퀀스 coreference resolution 능력을 최대한 끌어올릴 수 있습니다.