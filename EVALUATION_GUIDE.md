# MLM v2 Checkpoint Evaluation Guide

이 문서는 MLM v2로 훈련된 체크포인트들을 Real@1, Real@5, LAMBADA@1 메트릭으로 평가하는 방법을 설명합니다.

## 평가 메트릭 설명

### Real@1 (Entity Coreference Top-1 Accuracy)
- **목적**: 개체 상호참조 해결의 top-1 정확도 측정
- **방법**:
  - Wikipedia와 KLUE 데이터에서 같은 명사가 2번 이상 나오는 텍스트 추출
  - 대명사가 없는 텍스트만 사용
  - 2번째 명사를 `[MASK]`로 마스킹
  - Top-1 예측이 정답 명사와 일치하는지 확인
- **점수 범위**: 0.0 ~ 1.0 (높을수록 좋음)

### Real@5 (Entity Coreference Top-5 Recall)
- **목적**: 개체 상호참조 해결의 top-5 재현율 측정
- **방법**: Real@1과 동일하나, top-5 예측 중에 정답이 있는지 확인
- **점수 범위**: 0.0 ~ 1.0 (높을수록 좋음)

### LAMBADA@1 (Language Modeling Accuracy)
- **목적**: 일반적인 언어 모델링 능력 측정
- **방법**: Ko-LAMBADA 데이터셋에서 마지막 단어 예측
- **점수 범위**: 0.0 ~ 1.0 (높을수록 좋음)

### 종합 점수 계산
```
Score = 0.4 × Real@1 + 0.3 × Real@5 + 0.3 × LAMBADA@1
```

## 현재 베스트 성능 (비교 기준)

**체크포인트**: `runs/combined_experiment/checkpoint-1600` (Entity+MLM, seq_len=2048)
- **Real@1**: 67.78%
- **Real@5**: 82.44%
- **LAMBADA@1**: 66.13%
- **종합 점수**: 0.6785

## 사용 방법

### 1. 단일 체크포인트 평가

```bash
# 자동 시퀀스 길이 감지
python scripts/evaluate_checkpoint.py \
    --checkpoint runs/mlm_v2_scratch_1536/checkpoint-655

# 시퀀스 길이 명시적 지정
python scripts/evaluate_checkpoint.py \
    --checkpoint runs/mlm_v2_scratch_1536/checkpoint-655 \
    --seq-len 1536

# 결과 저장 디렉토리 지정
python scripts/evaluate_checkpoint.py \
    --checkpoint runs/mlm_v2_scratch_1536/checkpoint-655 \
    --seq-len 1536 \
    --output-dir ./my_evaluation_results
```

### 2. 배치 평가 (모든 체크포인트)

#### 방법 A: 자동 배치 스크립트 사용

```bash
# 모든 MLM v2 체크포인트 자동 평가
bash scripts/evaluate_all_mlm_v2_checkpoints.sh
```

이 스크립트는:
- `runs/mlm_v2_scratch_1536/` 의 모든 체크포인트 평가 (seq_len=1536)
- `runs/mlm_v2_scratch_2048/` 의 모든 체크포인트 평가 (seq_len=2048)
- 타임스탬프가 포함된 디렉토리에 결과 저장
- 진행 상황과 통계 표시

#### 방법 B: 수동 for 루프 사용

**1536 체크포인트:**
```bash
# 모든 1536 체크포인트 평가
for ckpt in runs/mlm_v2_scratch_1536/checkpoint-*; do
    echo "Evaluating $ckpt..."
    python scripts/evaluate_checkpoint.py --checkpoint "$ckpt" --seq-len 1536
done
```

**2048 체크포인트:**
```bash
# 모든 2048 체크포인트 평가
for ckpt in runs/mlm_v2_scratch_2048/checkpoint-*; do
    echo "Evaluating $ckpt..."
    python scripts/evaluate_checkpoint.py --checkpoint "$ckpt" --seq-len 2048
done
```

**특정 체크포인트 범위만 평가:**
```bash
# checkpoint-300 이상만 평가
for ckpt in runs/mlm_v2_scratch_1536/checkpoint-*; do
    checkpoint_num=$(basename "$ckpt" | sed 's/checkpoint-//')
    if [ "$checkpoint_num" -ge 300 ]; then
        python scripts/evaluate_checkpoint.py --checkpoint "$ckpt" --seq-len 1536
    fi
done
```

### 3. 결과 분석

#### JSON 결과 파일 보기
```bash
# 예쁘게 출력
cat runs/mlm_v2_scratch_1536/checkpoint-655_eval_results.json | jq '.'

# 주요 메트릭만 추출
cat runs/mlm_v2_scratch_1536/checkpoint-655_eval_results.json | jq '{checkpoint, real1, real5, lambada_top1}'
```

#### 모든 결과 요약
```bash
# 모든 결과 파일에서 Real@1, Real@5 추출
find runs/mlm_v2_scratch_* -name "*_eval_results.json" | while read file; do
    echo "=== $(basename $(dirname $file))/$(basename $file) ==="
    jq '{checkpoint, real1, real5, lambada_top1}' "$file"
    echo ""
done
```

#### 최고 성능 체크포인트 찾기
```bash
# Real@1 기준 정렬
find runs/mlm_v2_scratch_* -name "*_eval_results.json" -exec jq -r '"\(.real1)\t\(.checkpoint)"' {} \; | sort -rn | head -10

# 종합 점수 기준 정렬
find runs/mlm_v2_scratch_* -name "*_eval_results.json" -exec jq -r '"(0.4 * \(.real1) + 0.3 * \(.real5) + 0.3 * \(.lambada_top1))\t\(.checkpoint)"' {} \; | sort -rn | head -10
```

## 현재 사용 가능한 체크포인트

### MLM v2 @ 1536 (13개)
```
runs/mlm_v2_scratch_1536/checkpoint-65
runs/mlm_v2_scratch_1536/checkpoint-130
runs/mlm_v2_scratch_1536/checkpoint-195
runs/mlm_v2_scratch_1536/checkpoint-260
runs/mlm_v2_scratch_1536/checkpoint-325
runs/mlm_v2_scratch_1536/checkpoint-390
runs/mlm_v2_scratch_1536/checkpoint-455
runs/mlm_v2_scratch_1536/checkpoint-520
runs/mlm_v2_scratch_1536/checkpoint-585
runs/mlm_v2_scratch_1536/checkpoint-650
runs/mlm_v2_scratch_1536/checkpoint-655
```

### MLM v2 @ 2048 (8개)
```
runs/mlm_v2_scratch_2048/checkpoint-55
runs/mlm_v2_scratch_2048/checkpoint-110
runs/mlm_v2_scratch_2048/checkpoint-165
runs/mlm_v2_scratch_2048/checkpoint-220
runs/mlm_v2_scratch_2048/checkpoint-275
runs/mlm_v2_scratch_2048/checkpoint-330
runs/mlm_v2_scratch_2048/checkpoint-385
runs/mlm_v2_scratch_2048/checkpoint-440
```

## 예상 실행 시간

- **단일 체크포인트 평가**: ~5-10분 (GPU 사용 시)
- **1536 전체 배치 평가**: ~65-130분 (13개 체크포인트)
- **2048 전체 배치 평가**: ~40-80분 (8개 체크포인트)

## 출력 예시

```
================================================================================
✅ 평가 결과
================================================================================
체크포인트: runs/mlm_v2_scratch_1536/checkpoint-655
시퀀스 길이: 1536
Coref 샘플: 1600

LAMBADA@1: 0.6850 (68.50%)
Real@1:    0.6912 (69.12%)
Real@5:    0.8356 (83.56%)

평가 시간: 287.5초

────────────────────────────────────────────────────────────────────────────────
📊 성능 비교
────────────────────────────────────────────────────────────────────────────────
이전 베스트 (checkpoint-1600, Entity+MLM, seq_len=2048):
  - Real@1: 67.78%
  - Real@5: 82.44%

현재 체크포인트 (seq_len=1536):
  - Real@1: 69.12%
  - Real@5: 83.56%

변화:
  - Real@1: +1.34%p
  - Real@5: +1.12%p

종합 스코어 (0.4*Real@1 + 0.3*Real@5 + 0.3*LAMBADA@1):
  - 이전: 0.6785
  - 현재: 0.6824
  - 변화: +0.39%p
================================================================================
```

## 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기를 줄이려면 evaluate_checkpoint.py의 batch_size=64를 32로 변경
# 또는 CPU로 실행 (느림):
CUDA_VISIBLE_DEVICES=-1 python scripts/evaluate_checkpoint.py --checkpoint ...
```

### 시퀀스 길이 감지 실패
```bash
# --seq-len을 명시적으로 지정
python scripts/evaluate_checkpoint.py --checkpoint ... --seq-len 1536
```

### safetensors 모듈 없음
```bash
pip install safetensors
```

## 다음 단계

1. **최고 성능 체크포인트 식별**: 배치 평가 후 최고 Real@1/Real@5 점수 확인
2. **Entity v2 fine-tuning**: 최고 성능 체크포인트를 기반으로 Entity v2 데이터셋으로 추가 학습
3. **성능 비교**: MLM v2 단독 vs Entity v2 fine-tuning 결과 비교

### Entity v2 Fine-tuning 명령어
```bash
# 최고 성능 MLM v2 체크포인트로 Entity 학습
python scripts/run_entity_coref_finetune_v2.py \
    --checkpoint runs/mlm_v2_scratch_1536/checkpoint-655 \
    --dataset prepared_datasets/entity_coref_v2_1536 \
    --epochs 5 \
    --output-dir runs/entity_v2_finetune_from_mlm_v2
```

## 관련 파일

- `scripts/evaluate_checkpoint.py` - 단일 체크포인트 평가 스크립트
- `scripts/evaluate_all_mlm_v2_checkpoints.sh` - 배치 평가 자동화 스크립트
- `scripts/run_entity_coref_finetune_v2.py` - Entity v2 fine-tuning 스크립트
- `coref_automl/tune.py` - 평가 함수 구현 (`eval_real_coref_top1`, `eval_real_coref_top5`)
