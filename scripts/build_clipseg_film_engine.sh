#!/usr/bin/env bash
# Build CLIPSeg FiLM decoder TensorRT engine (FP16, DLA by default).
#   bash scripts/build_clipseg_film_engine.sh [data_dir] [model] [res] [batch] [num_prompts]
#
# Env: USE_DLA=1 (default), DLA_CORE=0|1 — see trt_common.sh
set -euo pipefail

HOUND_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=trt_common.sh
source "${HOUND_SCRIPTS}/trt_common.sh"

DATA="${1:-${HOUND_SCRIPTS}/../data}"
MODEL="${2:-CIDAS/clipseg-rd16}"
RES="${3:-224}"
BATCH="${4:-1}"
PROMPTS="${5:-9}"
STEM="clipseg_film_rd16_${RES}_b${BATCH}_p${PROMPTS}"
ONNX="${DATA}/${STEM}.onnx"
ENGINE="${DATA}/${STEM}.engine"

mkdir -p "${DATA}"

echo "=== 1) Export ONNX (${MODEL} @ ${RES} batch=${BATCH} prompts=${PROMPTS}) ==="
python3 "${HOUND_SCRIPTS}/export_clipseg_film_onnx.py" \
  --model "${MODEL}" \
  --input-res "${RES}" \
  --batch "${BATCH}" \
  --num-prompts "${PROMPTS}" \
  --output "${ONNX}"

echo "=== 2) Build TensorRT engine (FP16) ==="
resolve_trtexec
configure_trt_dla_flags static

"${TRTEXEC}" \
  --onnx="${ONNX}" \
  --saveEngine="${ENGINE}" \
  --fp16 \
  "${TRT_DLA_FLAGS[@]}" \
  --skipInference

echo "=== Done ==="
echo "ONNX:   ${ONNX}"
echo "Engine: ${ENGINE}"
