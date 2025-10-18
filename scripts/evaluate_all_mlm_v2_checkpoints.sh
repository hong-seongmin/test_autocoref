#!/bin/bash
# Batch evaluation script for all MLM v2 checkpoints with progress tracking and ETA
# Usage: bash scripts/evaluate_all_mlm_v2_checkpoints.sh

set -e

echo "================================================================================"
echo "🚀 MLM v2 Checkpoint Batch Evaluation"
echo "================================================================================"
echo ""

# Output directory for all results
OUTPUT_DIR="./evaluation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Results will be saved to: $OUTPUT_DIR"
echo ""

# Track timing
OVERALL_START=$(date +%s)
declare -a CHECKPOINT_TIMES

# Function to format seconds to HH:MM:SS
format_time() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

# Function to count checkpoints
count_checkpoints() {
    local base_dir=$1
    local count=0
    for ckpt in "$base_dir"/checkpoint-*; do
        if [ -d "$ckpt" ]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

# Count total checkpoints
TOTAL_CHECKPOINTS=0
if [ -d "runs/mlm_v2_scratch_1536" ]; then
    COUNT_1536=$(count_checkpoints "runs/mlm_v2_scratch_1536")
    echo "✓ Found $COUNT_1536 checkpoints in mlm_v2_scratch_1536"
    TOTAL_CHECKPOINTS=$((TOTAL_CHECKPOINTS + COUNT_1536))
fi
if [ -d "runs/mlm_v2_scratch_2048" ]; then
    COUNT_2048=$(count_checkpoints "runs/mlm_v2_scratch_2048")
    echo "✓ Found $COUNT_2048 checkpoints in mlm_v2_scratch_2048"
    TOTAL_CHECKPOINTS=$((TOTAL_CHECKPOINTS + COUNT_2048))
fi

echo ""
echo "📊 Total checkpoints to evaluate: $TOTAL_CHECKPOINTS"
echo "================================================================================"
echo ""

# Track progress
CURRENT_CHECKPOINT=0

# Function to evaluate checkpoints
evaluate_checkpoints() {
    local base_dir=$1
    local seq_len=$2
    local checkpoint_count=0

    echo ""
    echo "================================================================================"
    echo "📂 Evaluating: $base_dir (seq_len=$seq_len)"
    echo "================================================================================"
    echo ""

    # Count checkpoints in this directory
    for ckpt in "$base_dir"/checkpoint-*; do
        if [ -d "$ckpt" ]; then
            checkpoint_count=$((checkpoint_count + 1))
        fi
    done

    # Evaluate each checkpoint
    local local_current=0
    for ckpt in "$base_dir"/checkpoint-*; do
        if [ -d "$ckpt" ]; then
            local_current=$((local_current + 1))
            CURRENT_CHECKPOINT=$((CURRENT_CHECKPOINT + 1))
            checkpoint_name=$(basename "$ckpt")

            # Calculate progress
            PROGRESS_PCT=$(awk "BEGIN {printf \"%.1f\", ($CURRENT_CHECKPOINT / $TOTAL_CHECKPOINTS) * 100}")

            # Calculate ETA
            if [ ${#CHECKPOINT_TIMES[@]} -gt 0 ]; then
                # Average time per checkpoint
                TOTAL_TIME=0
                for t in "${CHECKPOINT_TIMES[@]}"; do
                    TOTAL_TIME=$((TOTAL_TIME + t))
                done
                AVG_TIME=$((TOTAL_TIME / ${#CHECKPOINT_TIMES[@]}))

                # Remaining checkpoints
                REMAINING=$((TOTAL_CHECKPOINTS - CURRENT_CHECKPOINT))
                ETA_SECONDS=$((AVG_TIME * REMAINING))
                ETA_FORMATTED=$(format_time $ETA_SECONDS)

                # Elapsed time
                CURRENT_TIME=$(date +%s)
                ELAPSED=$((CURRENT_TIME - OVERALL_START))
                ELAPSED_FORMATTED=$(format_time $ELAPSED)

                echo "────────────────────────────────────────────────────────────────────────────────"
                echo "🔍 [$CURRENT_CHECKPOINT/$TOTAL_CHECKPOINTS] ($PROGRESS_PCT%) Evaluating: $checkpoint_name"
                echo "📊 Progress: [$local_current/$checkpoint_count] in current directory"
                echo "⏱️  Elapsed: $ELAPSED_FORMATTED | ETA: $ETA_FORMATTED (avg: $(format_time $AVG_TIME)/checkpoint)"
                echo "────────────────────────────────────────────────────────────────────────────────"
            else
                echo "────────────────────────────────────────────────────────────────────────────────"
                echo "🔍 [$CURRENT_CHECKPOINT/$TOTAL_CHECKPOINTS] ($PROGRESS_PCT%) Evaluating: $checkpoint_name"
                echo "📊 Progress: [$local_current/$checkpoint_count] in current directory"
                echo "⏱️  First checkpoint - calculating ETA after completion..."
                echo "────────────────────────────────────────────────────────────────────────────────"
            fi

            # Time this checkpoint
            CHECKPOINT_START=$(date +%s)

            python scripts/evaluate_checkpoint.py \
                --checkpoint "$ckpt" \
                --seq-len "$seq_len" \
                --output-dir "$OUTPUT_DIR"

            CHECKPOINT_END=$(date +%s)
            CHECKPOINT_DURATION=$((CHECKPOINT_END - CHECKPOINT_START))
            CHECKPOINT_TIMES+=($CHECKPOINT_DURATION)

            echo ""
            echo "✅ Checkpoint completed in $(format_time $CHECKPOINT_DURATION)"
            echo ""
        fi
    done
}

# Evaluate 1536 checkpoints
if [ -d "runs/mlm_v2_scratch_1536" ]; then
    evaluate_checkpoints "runs/mlm_v2_scratch_1536" 1536
else
    echo "⚠️  runs/mlm_v2_scratch_1536 not found, skipping..."
fi

# Evaluate 2048 checkpoints
if [ -d "runs/mlm_v2_scratch_2048" ]; then
    evaluate_checkpoints "runs/mlm_v2_scratch_2048" 2048
else
    echo "⚠️  runs/mlm_v2_scratch_2048 not found, skipping..."
fi

# Final summary
OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
OVERALL_FORMATTED=$(format_time $OVERALL_DURATION)

# Calculate average time
if [ ${#CHECKPOINT_TIMES[@]} -gt 0 ]; then
    TOTAL_TIME=0
    for t in "${CHECKPOINT_TIMES[@]}"; do
        TOTAL_TIME=$((TOTAL_TIME + t))
    done
    AVG_TIME=$((TOTAL_TIME / ${#CHECKPOINT_TIMES[@]}))
    AVG_FORMATTED=$(format_time $AVG_TIME)
fi

echo ""
echo "================================================================================"
echo "✅ All evaluations complete!"
echo "================================================================================"
echo "📊 Statistics:"
echo "   - Total checkpoints evaluated: $TOTAL_CHECKPOINTS"
echo "   - Total time: $OVERALL_FORMATTED"
echo "   - Average time per checkpoint: $AVG_FORMATTED"
echo ""
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "📈 Quick analysis commands:"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""
echo "# View all results summary:"
echo "  cat $OUTPUT_DIR/*.json | jq -s 'sort_by(.real1) | reverse | .[] | {checkpoint, real1, real5, lambada_top1}'"
echo ""
echo "# Find best Real@1:"
echo "  cat $OUTPUT_DIR/*.json | jq -s 'sort_by(.real1) | reverse | .[0]'"
echo ""
echo "# Find best Real@5:"
echo "  cat $OUTPUT_DIR/*.json | jq -s 'sort_by(.real5) | reverse | .[0]'"
echo ""
echo "# Calculate and sort by composite score:"
echo "  cat $OUTPUT_DIR/*.json | jq -s 'map({checkpoint, score: (0.4 * .real1 + 0.3 * .real5 + 0.3 * .lambada_top1), real1, real5, lambada_top1}) | sort_by(.score) | reverse'"
echo ""
echo "================================================================================"
