#!/usr/bin/env bash
# ============================================================
#  Tool-Reflection-Bench — One-Click GRPO Training
# ============================================================
set -euo pipefail

# ---------- defaults ----------
MODEL=""
MODEL_TYPE=""
REWARD_FUNC=""
TRAIN_DATA=""
OUTPUT_DIR=""
NUM_GPUS=8

usage() {
    cat <<EOF
Usage: bash train.sh --model <model> [options]

Required:
  --model <name|path>   One of: qwen2.5-7b, qwen3-4b, llama3.1-8b,
                        or a full path to a HuggingFace model directory.

Options:
  --data <path>         Training data jsonl (default: auto-select based on model)
  --output <dir>        Output directory (default: ../outputs/<model_name>-GRPO)
  --gpus <n>            Number of GPUs (default: 8)
  -h, --help            Show this help

Examples:
  bash train.sh --model qwen2.5-7b
  bash train.sh --model /path/to/Qwen2.5-7B-Instruct --data ../data/train_qwen.jsonl
  bash train.sh --model llama3.1-8b --gpus 4
EOF
    exit 0
}

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2;;
        --data)    TRAIN_DATA="$2"; shift 2;;
        --output)  OUTPUT_DIR="$2"; shift 2;;
        --gpus)    NUM_GPUS="$2"; shift 2;;
        -h|--help) usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done

[[ -z "$MODEL" ]] && { echo "Error: --model is required"; usage; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------- resolve model shorthand ----------
resolve_model() {
    local m="$1"
    case "${m,,}" in
        qwen2.5-7b|qwen2.5)
            MODEL_TYPE="qwen2_5"
            REWARD_FUNC="ChainDialogueRewardOptimized_qwen"
            # User must have the model locally; use HF ID as hint
            [[ ! -d "$MODEL" ]] && echo "[WARN] Model path '$MODEL' not found; download Qwen/Qwen2.5-7B-Instruct first."
            ;;
        qwen3-4b|qwen3)
            MODEL_TYPE="qwen3"
            REWARD_FUNC="ChainDialogueRewardOptimized_qwen3"
            [[ ! -d "$MODEL" ]] && echo "[WARN] Model path '$MODEL' not found; download Qwen/Qwen3-4B-Instruct first."
            ;;
        llama3.1-8b|llama3.1|llama)
            MODEL_TYPE="llama3_1"
            REWARD_FUNC="ChainDialogueRewardOptimized_llama"
            [[ ! -d "$MODEL" ]] && echo "[WARN] Model path '$MODEL' not found; download meta-llama/Llama-3.1-8B-Instruct first."
            ;;
        *)
            # Assume it's a full path; auto-detect type from path name
            if echo "$m" | grep -qi "qwen3"; then
                MODEL_TYPE="qwen3"; REWARD_FUNC="ChainDialogueRewardOptimized_qwen3"
            elif echo "$m" | grep -qi "qwen"; then
                MODEL_TYPE="qwen2_5"; REWARD_FUNC="ChainDialogueRewardOptimized_qwen"
            elif echo "$m" | grep -qi "llama"; then
                MODEL_TYPE="llama3_1"; REWARD_FUNC="ChainDialogueRewardOptimized_llama"
            else
                echo "Error: Cannot detect model type from '$m'. Use --model qwen2.5-7b|qwen3-4b|llama3.1-8b"
                exit 1
            fi
            ;;
    esac
}

resolve_model "$MODEL"

# ---------- auto-select training data ----------
if [[ -z "$TRAIN_DATA" ]]; then
    if [[ "$MODEL_TYPE" == "llama3_1" ]]; then
        TRAIN_DATA="$PROJECT_DIR/data/train_llama.jsonl"
    else
        TRAIN_DATA="$PROJECT_DIR/data/train_qwen.jsonl"
    fi
fi

[[ ! -f "$TRAIN_DATA" ]] && { echo "Error: Training data not found: $TRAIN_DATA"; exit 1; }
SAMPLE_COUNT=$(wc -l < "$TRAIN_DATA")

# ---------- auto-select output dir ----------
MODEL_BASENAME=$(basename "$MODEL")
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROJECT_DIR/outputs/${MODEL_BASENAME}-GRPO"
fi

# ---------- environment ----------
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

# ---------- patch training script ----------
# Create a temp copy with model_type and reward_funcs replaced
TRAIN_SCRIPT=$(mktemp /tmp/train_XXXXXX.py)
cp "$SCRIPT_DIR/grpo_training_script.py" "$TRAIN_SCRIPT"

sed -i "s|model_type='qwen2_5'|model_type='${MODEL_TYPE}'|g" "$TRAIN_SCRIPT"
sed -i "s|reward_funcs=\['ChainDialogueRewardOptimized_qwen'\]|reward_funcs=['${REWARD_FUNC}']|g" "$TRAIN_SCRIPT"

# ---------- summary ----------
echo "============================================"
echo " Tool-Reflection-Bench GRPO Training"
echo "============================================"
echo " Model:        $MODEL"
echo " Model type:   $MODEL_TYPE"
echo " Reward func:  $REWARD_FUNC"
echo " Dataset:      $TRAIN_DATA ($SAMPLE_COUNT samples)"
echo " Output:       $OUTPUT_DIR"
echo " GPUs:         $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

# ---------- launch ----------
python -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    "$TRAIN_SCRIPT" \
    --model_dir "$MODEL" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Training complete. Output: $OUTPUT_DIR"
