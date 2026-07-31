#!/usr/bin/env bash
# Build CLIPSeg vision TensorRT engine (FP16) for Orin.
#
# Run inside mushr_jazzy (GPU + trtexec):
#   bash /root/colcon_ws/src/perception_models/scripts/build_clipseg_vision_engine.sh
#
# Then A/B:
#   python3 .../bench_clipseg.py --trt-engine .../clipseg_vision_rd16_224.engine
set -euo pipefail

HOUND_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
# Default: perception_models/data next to this package
DATA="${1:-${HOUND_SCRIPTS}/../data}"
MODEL="${2:-CIDAS/clipseg-rd16}"
RES="${3:-224}"
STEM="clipseg_vision_rd16_${RES}"
ONNX="${DATA}/${STEM}.onnx"
ENGINE="${DATA}/${STEM}.engine"

mkdir -p "${DATA}"

echo "=== 1) Export ONNX (${MODEL} @ ${RES}) ==="
python3 "${HOUND_SCRIPTS}/export_clipseg_vision_onnx.py" \
  --model "${MODEL}" \
  --input-res "${RES}" \
  --output "${ONNX}"

echo "=== 2) Build TensorRT engine (FP16) ==="
# Prefer /usr/src/tensorrt/bin/trtexec on Jetson images.
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
echo "Bench:"
echo "  python3 ${HOUND_SCRIPTS}/bench_clipseg.py --trt-engine ${ENGINE}"
