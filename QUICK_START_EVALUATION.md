# Quick Start: MLM v2 Checkpoint Evaluation

빠른 평가를 위한 간단 가이드입니다.

## 🚀 단일 체크포인트 평가 (5분)

```bash
# 가장 최근 1536 체크포인트 평가
python scripts/evaluate_checkpoint.py \
    --checkpoint runs/mlm_v2_scratch_1536/checkpoint-655

# 가장 최근 2048 체크포인트 평가
python scripts/evaluate_checkpoint.py \
    --checkpoint runs/mlm_v2_scratch_2048/checkpoint-440
```

## 🔄 모든 체크포인트 배치 평가 (1-2시간)

```bash
# 한 번의 명령으로 모든 MLM v2 체크포인트 평가
bash scripts/evaluate_all_mlm_v2_checkpoints.sh
```

## 📊 평가 메트릭

| 메트릭 | 설명 | 현재 베스트 |
|--------|------|-------------|
| **Real@1** | 개체 상호참조 Top-1 정확도 | 67.78% |
| **Real@5** | 개체 상호참조 Top-5 재현율 | 82.44% |
| **LAMBADA@1** | 언어 모델링 정확도 | 66.13% |

**종합 점수**: `0.4 × Real@1 + 0.3 × Real@5 + 0.3 × LAMBADA@1` = **0.6785**

## 📁 체크포인트 현황

### 1536 시퀀스 (13개)
- checkpoint-65, 130, 195, 260, 325, 390, 455, 520, 585, 650, **655**

### 2048 시퀀스 (8개)
- checkpoint-55, 110, 165, 220, 275, 330, 385, **440**

## 🎯 다음 단계

### 1. 최고 성능 체크포인트 찾기
```bash
# Real@1 기준 Top 5
find runs/mlm_v2_scratch_* -name "*_eval_results.json" -exec jq -r '"\(.real1)\t\(.checkpoint)"' {} \; | sort -rn | head -5
```

### 2. Entity v2 Fine-tuning
```bash
# 최고 성능 체크포인트로 Entity 학습
python scripts/run_entity_coref_finetune_v2.py \
    --checkpoint runs/mlm_v2_scratch_1536/checkpoint-655 \
    --dataset prepared_datasets/entity_coref_v2_1536 \
    --epochs 5
```

## 💡 팁

- **빠른 테스트**: 최신 체크포인트 1-2개만 평가해서 경향 파악
- **배치 실행**: 전체 평가는 백그라운드로 실행 (`nohup bash scripts/evaluate_all_mlm_v2_checkpoints.sh &`)
- **결과 비교**: 여러 체크포인트를 평가한 후 학습 곡선 분석

## 📖 자세한 정보

전체 가이드: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
