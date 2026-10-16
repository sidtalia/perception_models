#!/usr/bin/env bash
# Install runtime deps for perception_models (Dora CLIPSeg / NanoSAM / YOLO-World).
# Skips mesh-ICP, FoundationStereo, spencer_port — those are out of scope here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_SRC="$(cd "${ROOT}/.." && pwd)"
NANOSAM_SRC="${NANOSAM_SRC:-${WS_SRC}/nanosam-src}"

echo "[perception_models] install_deps from ${ROOT}"

python3 -m pip install --upgrade pip
# Dora node API + CLI (provides `dora` binary)
python3 -m pip install 'dora-rs-cli>=0.3.0' 'dora-rs>=0.3.0' pyarrow

# Model stacks (torch / CUDA usually preinstalled on Jetson images)
python3 -m pip install 'transformers>=4.36' ultralytics

if [[ -d "${NANOSAM_SRC}" ]]; then
  echo "[perception_models] pip install -e ${NANOSAM_SRC}"
  python3 -m pip install -e "${NANOSAM_SRC}"
else
  echo "[perception_models] WARNING: nanosam-src not at ${NANOSAM_SRC}"
  echo "  Clone https://github.com/NVIDIA-AI-IOT/nanosam and set NANOSAM_SRC=..."
fi

# TensorRT Python + torch2trt are environment-specific on Jetson; do not force pip.
if python3 -c "import tensorrt" 2>/dev/null; then
  echo "[perception_models] tensorrt: OK"
else
  echo "[perception_models] NOTE: import tensorrt failed — install Jetson TensorRT packages for TRT engines"
fi
if python3 -c "import torch2trt" 2>/dev/null; then
  echo "[perception_models] torch2trt: OK"
else
  echo "[perception_models] NOTE: torch2trt not importable — needed only for some TRT conversion paths"
fi

echo "[perception_models] editable package"
python3 -m pip install -e "${ROOT}" || true

echo "[perception_models] done. Rebuild engines on this device:"
echo "  bash ${ROOT}/scripts/build_all_engines.sh"
