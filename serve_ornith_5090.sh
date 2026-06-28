#!/usr/bin/env bash
#
# serve_ornith_5090.sh - local vLLM launcher for single or dual RTX 5090 systems.
#
# This is the consumer Blackwell path, not the DGX Spark Docker/DFlash path.
# It serves the NVFP4 build fully on GPU with no CPU offload.
#
# Usage:
#   ./serve_ornith_5090.sh single
#   ./serve_ornith_5090.sh dual
#
# Environment overrides:
#   MODEL, SERVED_NAME, PORT, MAX_LEN, GPU_MEM, MAX_NUM_SEQS,
#   MAX_BATCHED_TOKENS, CUDA_VISIBLE_DEVICES

set -euo pipefail

PROFILE="${1:-dual}"

MODEL="${MODEL:-AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4}"
SERVED_NAME="${SERVED_NAME:-ornith}"
PORT="${PORT:-8000}"

case "$PROFILE" in
  single)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    TP_SIZE="${TP_SIZE:-1}"
    MAX_LEN="${MAX_LEN:-32768}"
    GPU_MEM="${GPU_MEM:-0.88}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
    ;;
  dual)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    TP_SIZE="${TP_SIZE:-2}"
    MAX_LEN="${MAX_LEN:-65536}"
    GPU_MEM="${GPU_MEM:-0.78}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
    MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
    ;;
  *)
    echo "unknown profile '$PROFILE' (use: single | dual)" >&2
    exit 2
    ;;
esac

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

echo "=== Ornith RTX 5090 vLLM launcher ==="
echo "  profile               : $PROFILE"
echo "  model                 : $MODEL"
echo "  served name           : $SERVED_NAME"
echo "  cuda visible devices  : $CUDA_VISIBLE_DEVICES"
echo "  tensor parallel size  : $TP_SIZE"
echo "  max model len         : $MAX_LEN"
echo "  gpu memory utilization: $GPU_MEM"
echo "  max num seqs          : $MAX_NUM_SEQS"
echo "  max batched tokens    : $MAX_BATCHED_TOKENS"
echo "  endpoint              : http://127.0.0.1:$PORT/v1"
echo "======================================"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --quantization compressed-tensors \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --mamba-cache-dtype float32 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT"
