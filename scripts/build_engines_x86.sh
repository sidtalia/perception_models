#!/usr/bin/env bash
# Build all prompted-segmentation engines on an x86 desktop GPU (pip TensorRT, no trtexec).
#
#   PY=/path/to/venv/bin/python bash scripts/build_engines_x86.sh
#
# SAM_ENCODER selects the SAM image encoder:
#   tinyvit  (default) MobileSAM TinyViT @1024, exported here from mobile_sam.pt.
#   resnet18           NanoSAM distilled ResNet18 @512. Needs data/resnet18_image_encoder.onnx
#                      (Google Drive only; set NANOSAM_ENCODER_ONNX to a local copy).
# Both feed the same batched MobileSAM mask decoder (exported at the matching size).
#
# Fetches ../nanosam-src (MobileSAM code + checkpoint) if missing; trtexec is the
# scripts/trtexec_py shim. Env: IMAGE_BATCH (1), CLIPSEG_FILM_PROMPTS (4),
# OPT_BATCH (8), MAX_BATCH (16), ENC_MAX_BATCH (4).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PY="${PY:-python3}"
PYBIN="$(dirname "$(command -v "${PY}")")"
export PATH="${PYBIN}:${HERE}:${PATH}"   # python3 -> venv; trtexec -> shim
ln -sf "${HERE}/trtexec_py" "${HERE}/trtexec"
export DATA="${DATA:-${ROOT}/data}"
mkdir -p "${DATA}"
SAM_ENCODER="${SAM_ENCODER:-tinyvit}"
IMAGE_BATCH="${IMAGE_BATCH:-1}"
OPT_BATCH="${OPT_BATCH:-8}"
MAX_BATCH="${MAX_BATCH:-16}"
ENC_MAX_BATCH="${ENC_MAX_BATCH:-4}"

NANOSAM_SRC="${NANOSAM_SRC:-${ROOT}/../nanosam-src}"
if [[ ! -d "${NANOSAM_SRC}" ]]; then
  git clone --depth 1 https://github.com/NVIDIA-AI-IOT/nanosam.git "${NANOSAM_SRC}"
fi
python3 -c "import nanosam" 2>/dev/null || \
  echo "${NANOSAM_SRC}" > "$(python3 -c 'import site;print(site.getsitepackages()[0])')/nanosam_src.pth"
mkdir -p "${NANOSAM_SRC}/assets"
if [[ ! -s "${NANOSAM_SRC}/assets/mobile_sam.pt" ]]; then
  curl -fL -o "${NANOSAM_SRC}/assets/mobile_sam.pt" \
    https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
fi
export NANOSAM_CKPT="${NANOSAM_SRC}/assets/mobile_sam.pt"

echo ">>> CLIPSeg vision + FiLM (image batch ${IMAGE_BATCH}, ${CLIPSEG_FILM_PROMPTS:-4} prompt slots)"
USE_DLA=0 USE_DLA_DYNAMIC=0 SKIP_YOLO=1 SKIP_NANOSAM=1 \
IMAGE_BATCH="${IMAGE_BATCH}" CLIPSEG_FILM_PROMPTS="${CLIPSEG_FILM_PROMPTS:-4}" \
  bash "${HERE}/build_all_engines.sh"

if [[ "${SAM_ENCODER}" == "resnet18" ]]; then
  if [[ ! -s "${DATA}/resnet18_image_encoder.onnx" ]]; then
    cp "${NANOSAM_ENCODER_ONNX:?set NANOSAM_ENCODER_ONNX to resnet18_image_encoder.onnx}" \
      "${DATA}/resnet18_image_encoder.onnx"
  fi
  USE_DLA=0 USE_DLA_DYNAMIC=0 SKIP_YOLO=1 SKIP_CLIPSEG=1 SKIP_CLIPSEG_FILM=1 \
  IMAGE_BATCH="${IMAGE_BATCH}" OPT_BATCH="${OPT_BATCH}" MAX_BATCH="${MAX_BATCH}" \
  ENC_OPT_BATCH="${IMAGE_BATCH}" ENC_MAX_BATCH="${ENC_MAX_BATCH}" \
    bash "${HERE}/build_all_engines.sh"
  exit 0
fi

SIZE=1024; EMB=64; MASK_IN=256
ENC_ONNX="${DATA}/mobilesam_tinyvit_${SIZE}_b${IMAGE_BATCH}.onnx"
ENC_ENGINE="${DATA}/mobilesam_tinyvit_${SIZE}_b${IMAGE_BATCH}.engine"
DEC_ONNX="${DATA}/mobile_sam_mask_decoder_batched_${SIZE}.onnx"
DEC_ENGINE="${DATA}/mobile_sam_mask_decoder_batched_${SIZE}.engine"

echo ">>> MobileSAM TinyViT encoder @${SIZE} (fixed batch ${IMAGE_BATCH})"
python3 "${HERE}/export_mobilesam_image_encoder_onnx.py" --checkpoint "${NANOSAM_CKPT}" \
  --batch "${IMAGE_BATCH}" --output "${ENC_ONNX}"
# FP32 on purpose: TinyViT LayerNorms overflow in FP16 under TensorRT 10.13 (embedding cos -0.24 vs eager).
trtexec --onnx="${ENC_ONNX}" --saveEngine="${ENC_ENGINE}" --skipInference

echo ">>> Batched MobileSAM mask decoder @${SIZE} (opt ${OPT_BATCH}, max ${MAX_BATCH})"
python3 "${HERE}/export_nanosam_mask_decoder_batched_onnx.py" \
  --checkpoint "${NANOSAM_CKPT}" --output "${DEC_ONNX}" --image-size "${SIZE}"
USE_DLA=0 USE_DLA_DYNAMIC=0 bash "${HERE}/build_nanosam_mask_decoder_batched_engine.sh" \
  "${DEC_ONNX}" "${DEC_ENGINE}" "${OPT_BATCH}" "${MAX_BATCH}" "${EMB}" "${EMB}" "${MASK_IN}"

echo ""
echo "Engines in ${DATA}:"
ls -lh "${DATA}"/*.engine
echo "Config keys:  sam_image_size: ${SIZE}  sam_image_encoder: ${ENC_ENGINE}  sam_mask_decoder: ${DEC_ENGINE}"
