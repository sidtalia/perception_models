#!/usr/bin/env bash
# Build CLIPSeg vision TensorRT engine (FP16).
#   bash scripts/build_clipseg_vision_engine.sh [data_dir] [model] [res] [batch]
set -euo pipefail

HOUND_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
DATA="${1:-${HOUND_SCRIPTS}/../data}"
MODEL="${2:-CIDAS/clipseg-rd16}"
RES="${3:-224}"
BATCH="${4:-1}"
STEM="clipseg_vision_rd16_${RES}"
if [[ "${BATCH}" -gt 1 ]]; then
  STEM="${STEM}_b${BATCH}"
fi
ONNX="${DATA}/${STEM}.onnx"
ENGINE="${DATA}/${STEM}.engine"

mkdir -p "${DATA}"

echo "=== 1) Export ONNX (${MODEL} @ ${RES} batch=${BATCH}) ==="
python3 "${HOUND_SCRIPTS}/export_clipseg_vision_onnx.py" \
  --model "${MODEL}" \
  --input-res "${RES}" \
  --batch "${BATCH}" \
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
