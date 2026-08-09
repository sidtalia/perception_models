#!/usr/bin/env bash
# Build CLIPSeg FiLM decoder TensorRT engine (FP16).
#   bash scripts/build_clipseg_film_engine.sh [data_dir] [model] [res] [batch] [num_prompts]
set -euo pipefail

HOUND_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
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
TRTEXEC="$(command -v trtexec || true)"
if [[ -z "${TRTEXEC}" && -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [[ -z "${TRTEXEC}" ]]; then
  echo "trtexec not found" >&2
  exit 1
fi

"${TRTEXEC}" \
  --onnx="${ONNX}" \
  --saveEngine="${ENGINE}" \
  --fp16 \
  --skipInference

echo "=== Done ==="
echo "ONNX:   ${ONNX}"
echo "Engine: ${ENGINE}"
