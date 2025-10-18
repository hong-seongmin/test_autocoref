#!/bin/bash
# 개선된 데이터셋으로 학습 실행
# 사용법: bash run_training.sh

echo "=========================================="
echo "Coreference Resolution Training"
echo "개선된 필터링 데이터셋 사용"
echo "=========================================="
echo ""

# 데이터셋 경로 확인
echo "📁 데이터셋 확인 중..."
datasets=()

if [ -d "prepared_datasets/klue_mrc_filtered_1536" ]; then
    datasets+=("--dataset-choice prepared_datasets/klue_mrc_filtered_1536")
    echo "  ✅ KLUE MRC (1536)"
fi

if [ -d "prepared_datasets/klue_mrc_filtered_2048" ]; then
    datasets+=("--dataset-choice prepared_datasets/klue_mrc_filtered_2048")
    echo "  ✅ KLUE MRC (2048)"
fi

if [ -d "prepared_datasets/wikipedia_filtered_1536" ]; then
    datasets+=("--dataset-choice prepared_datasets/wikipedia_filtered_1536")
    echo "  ✅ Wikipedia (1536)"
fi

if [ -d "prepared_datasets/wikipedia_filtered_2048" ]; then
    datasets+=("--dataset-choice prepared_datasets/wikipedia_filtered_2048")
    echo "  ✅ Wikipedia (2048)"
fi

if [ -d "prepared_datasets/naver_news_filtered_1536" ]; then
    datasets+=("--dataset-choice prepared_datasets/naver_news_filtered_1536")
    echo "  ✅ Naver News (1536)"
fi

if [ -d "prepared_datasets/naver_news_filtered_2048" ]; then
    datasets+=("--dataset-choice prepared_datasets/naver_news_filtered_2048")
    echo "  ✅ Naver News (2048)"
fi

if [ ${#datasets[@]} -eq 0 ]; then
    echo "❌ 데이터셋이 없습니다!"
    echo "먼저 다음 명령어로 데이터셋을 생성하세요:"
    echo "  python scripts/prepare_filtered_datasets.py"
    exit 1
fi

echo ""
echo "📊 총 ${#datasets[@]}개 데이터셋 발견"
echo ""

# 학습 설정
MODEL="kakaobank/kf-deberta-base"
SEQ_LENGTHS="1536 2048"
TRIALS=30  # Optuna trial 수
EPOCHS="2 3"  # 에폭 선택지

echo "⚙️  학습 설정:"
echo "  모델: $MODEL"
echo "  시퀀스 길이: $SEQ_LENGTHS"
echo "  Optuna Trials: $TRIALS"
echo "  에폭 선택지: $EPOCHS"
echo ""

# 명령어 생성
CMD="python -m coref_automl.long_sequence_automl \
    --model $MODEL \
    --seq-lengths $SEQ_LENGTHS \
    --trials $TRIALS \
    --epoch-choices $EPOCHS"

# 데이터셋 추가
for dataset in "${datasets[@]}"; do
    CMD="$CMD $dataset"
done

echo "🚀 학습 시작..."
echo ""
echo "실행 명령어:"
echo "$CMD"
echo ""
echo "=========================================="
echo ""

# 실행
eval $CMD
