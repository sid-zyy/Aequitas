#!/usr/bin/env bash
# Serves a Gemma 4 model with vLLM for Aequitas, in one of the benchmark profiles.
#
#   MODEL=google/gemma-4-E2B-it QUANT=fp8 SPEC=mtp bash scripts/vllm/serve.sh
#
# QUANT   bf16   | the original weights (E2B is ~10 GB in BF16: on a 6 GB GPU this needs CPU_OFFLOAD_GB)
#         fp8    | vLLM's online dynamic FP8 (no calibration; needs compute capability >= 8.9)
#         <path> | a checkpoint from scripts/quantize (W8A8 / W4A16 / FP8, compressed-tensors, auto-detected)
# SPEC    none | mtp (Gemma 4 MTP drafter = the -assistant checkpoint) | ngram (prompt lookup, for chat)
# NSPEC   speculative tokens per step (default: 1 for mtp, as in vLLM's docs; 4 for ngram). Sweep it.
#
# Then in Aequitas's config:  vlm: {backend: openai, host: http://127.0.0.1:${PORT}, model: ${SERVED_NAME}}
# Check flag names against `vllm serve --help` for the installed version before trusting a result.
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E2B-it}"
QUANT="${QUANT:-fp8}"
SPEC="${SPEC:-none}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-gemma4}"
MAX_LEN="${MAX_LEN:-4096}"            # one 2x2 grid + prompt + short output fits easily
GPU_UTIL="${GPU_UTIL:-0.85}"          # leave room for the Windows desktop on a laptop GPU
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
MAX_SOFT_TOKENS="${MAX_SOFT_TOKENS:-280}"  # Gemma 4 vision token budget (280-1120): fewer = faster prefill
DRAFTER="${DRAFTER:-${MODEL%-it}-it-assistant}"  # google/gemma-4-E2B-it -> google/gemma-4-E2B-it-assistant

args=(serve "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0 --port "$PORT"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --limit-mm-per-prompt '{"image":1,"audio":0}'
  --enable-prefix-caching
)

case "$QUANT" in
  bf16) args+=(--dtype bfloat16) ;;
  fp8)  args+=(--quantization fp8) ;;
  *)    [ -d "$QUANT" ] || { echo "QUANT must be bf16, fp8 or a checkpoint directory"; exit 1; }
        args[1]="$QUANT" ;;  # serve the quantized checkpoint; its config says how it is quantized
esac
[ "$CPU_OFFLOAD_GB" != "0" ] && args+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
# Gemma-only option: Qwen3-VL and SmolVLM2 reject it
if [[ "${MODEL,,} ${QUANT,,}" == *gemma* ]]; then
  args+=(--mm-processor-kwargs "{\"max_soft_tokens\":${MAX_SOFT_TOKENS}}")
fi
[ -n "${DTYPE:-}" ] && args+=(--dtype "$DTYPE")  # e.g. half on GPUs without BF16

case "$SPEC" in
  none)  ;;
  mtp)   args+=(--speculative-config "{\"method\":\"mtp\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${NSPEC:-1}}") ;;
  ngram) args+=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${NSPEC:-4},\"prompt_lookup_max\":${LOOKUP_MAX:-4}}") ;;
  *)     echo "SPEC must be none, mtp or ngram"; exit 1 ;;
esac

echo "vllm ${args[*]}"
exec vllm "${args[@]}"
