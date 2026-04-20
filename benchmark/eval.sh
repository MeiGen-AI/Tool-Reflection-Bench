#!/usr/bin/env bash
# ============================================================
#  Tool-Reflection-Bench — Evaluation Script
# ============================================================
set -euo pipefail

usage() {
    cat <<EOF
Usage:
  Local model (vLLM):
    bash eval.sh --model <path> [--n N] [--tp TP] [--batch B] [--gpu-mem F]

  API model:
    bash eval.sh --api <model_name> [--n N] [--api-key KEY] [--base-url URL]

Options:
  --model <path>      Path to local HuggingFace model directory
  --api <name>        API model name (e.g., gpt-4o, Qwen/Qwen2.5-72B-Instruct)
  --n <int>           Pass@N attempts per question (default: 1)
  --tp <int>          Tensor parallel size for vLLM (default: auto)
  --batch <int>       Batch size for vLLM (default: 64)
  --gpu-mem <float>   GPU memory utilization (default: 0.9)
  --api-key <str>     API key (default: \$OPENAI_API_KEY)
  --base-url <str>    API base URL (default: \$OPENAI_BASE_URL)
  -h, --help          Show this help

Examples:
  bash eval.sh --model /path/to/Qwen2.5-7B-Instruct --n 5 --tp 4
  bash eval.sh --api gpt-4o --n 3
  bash eval.sh --api Qwen/Qwen2.5-72B-Instruct --base-url https://your-openai-compatible-endpoint/v1
EOF
    exit 0
}

# ---------- defaults ----------
MODEL="" API="" N=1 TP="" BATCH=64 GPU_MEM=0.9 API_KEY="" BASE_URL=""

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2"; shift 2;;
        --api)      API="$2"; shift 2;;
        --n)        N="$2"; shift 2;;
        --tp)       TP="$2"; shift 2;;
        --batch)    BATCH="$2"; shift 2;;
        --gpu-mem)  GPU_MEM="$2"; shift 2;;
        --api-key)  API_KEY="$2"; shift 2;;
        --base-url) BASE_URL="$2"; shift 2;;
        -h|--help)  usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done

[[ -z "$MODEL" && -z "$API" ]] && { echo "Error: specify --model or --api"; usage; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------- build python args ----------
PY_ARGS=("--n" "$N" "--batch" "$BATCH" "--gpu-mem" "$GPU_MEM")

if [[ -n "$MODEL" ]]; then
    PY_ARGS+=("--model" "$MODEL")
    MODEL_SHORT=$(basename "$MODEL")
    [[ -n "$TP" ]] && PY_ARGS+=("--tp" "$TP")
    echo "============================================"
    echo " Tool-Reflection-Bench Evaluation"
    echo "============================================"
    echo " Mode:     vLLM (local)"
    echo " Model:    $MODEL_SHORT"
    echo " Path:     $MODEL"
    echo " TP:       ${TP:-auto}"
    echo " GPU mem:  $GPU_MEM"
    echo " Batch:    $BATCH"
    echo " Pass@N:   N=$N"
    echo "============================================"
else
    PY_ARGS+=("--api" "$API")
    MODEL_SHORT="$API"
    [[ -n "$API_KEY" ]] && PY_ARGS+=("--api-key" "$API_KEY")
    [[ -n "$BASE_URL" ]] && PY_ARGS+=("--base-url" "$BASE_URL")
    echo "============================================"
    echo " Tool-Reflection-Bench Evaluation"
    echo "============================================"
    echo " Mode:     API"
    echo " Model:    $API"
    echo " Pass@N:   N=$N"
    echo "============================================"
fi

echo ""

# ---------- set PYTHONPATH for swift imports if needed ----------
export PYTHONPATH="$PROJECT_DIR/train:${PYTHONPATH:-}"

# ---------- run ----------
python "$SCRIPT_DIR/test.py" "${PY_ARGS[@]}"
