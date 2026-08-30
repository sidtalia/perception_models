#!/usr/bin/env bash
# Build all HOUND perception model artifacts on *this* GPU box (Orin NX / Thor).
#
# TensorRT engines must be rebuilt per JetPack/TRT/GPU — do not copy .engine
# files between machines. ONNX / stashed .pt can be reused as inputs.
#
# Inside mushr_jazzy (or Thor container with trtexec + CUDA):
#   bash /root/colcon_ws/src/perception_models/scripts/build_all_engines.sh
#
# Default profile (3-cam SSoT): CLIPSeg vision+FiLM image batch 3, 9 FiLM
# prompts; NanoSAM encoder 512 dynamic batch opt/max 3; decoder opt 16 max 32.
#
# Env overrides:
#   DATA=...          output dir (default: perception_models/data)
#   IMAGE_BATCH=3     CLIPSeg vision/FiLM N and SAM encoder TRT opt/max batch
#   CLIPSEG_FILM_PROMPTS=9
#   OPT_BATCH=16      NanoSAM decoder TRT opt batch
#   MAX_BATCH=32      NanoSAM decoder TRT max batch (sam_max_boxes* <= this)
#   USE_DLA=1         Build for Jetson DLA (+ GPU fallback). Default 1.
#   USE_DLA_DYNAMIC=1 DLA flags on dynamic-shape NanoSAM engines too. Default 1.
#   SKIP_CLIPSEG=1    skip CLIPSeg vision engine
#   SKIP_NANOSAM=1    skip NanoSAM 512 encoder+decoder pair
#   SKIP_YOLO=1       skip YOLO-World stashed .pt (torch path; no TRT yet)
#   YOLO_MODEL=...    ultralytics weights (default: yolov8s-world.pt)
#   YOLO_CLASSES=...  space-separated class list
#
# DLA core assignment (Orin has 0 and 1) so rebuilds are deterministic:
#   CLIPSeg vision + NanoSAM encoder → DLA 0
#   CLIPSeg FiLM  + NanoSAM decoder  → DLA 1
set -euo pipefail

HOUND_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-${HOUND_SCRIPTS}/../data}"
IMAGE_BATCH="${IMAGE_BATCH:-3}"
CLIPSEG_FILM_BATCH="${CLIPSEG_FILM_BATCH:-${IMAGE_BATCH}}"
CLIPSEG_FILM_PROMPTS="${CLIPSEG_FILM_PROMPTS:-9}"
ENC_OPT_BATCH="${ENC_OPT_BATCH:-${IMAGE_BATCH}}"
ENC_MAX_BATCH="${ENC_MAX_BATCH:-${IMAGE_BATCH}}"
OPT_BATCH="${OPT_BATCH:-16}"
MAX_BATCH="${MAX_BATCH:-32}"
USE_DLA="${USE_DLA:-1}"
USE_DLA_DYNAMIC="${USE_DLA_DYNAMIC:-1}"
SKIP_CLIPSEG="${SKIP_CLIPSEG:-0}"
SKIP_CLIPSEG_FILM="${SKIP_CLIPSEG_FILM:-0}"
SKIP_NANOSAM="${SKIP_NANOSAM:-0}"
SKIP_YOLO="${SKIP_YOLO:-0}"
YOLO_MODEL="${YOLO_MODEL:-yolov8s-world.pt}"
# shellcheck disable=SC2206
YOLO_CLASSES=(${YOLO_CLASSES:-person chair door table bag})

export USE_DLA USE_DLA_DYNAMIC

mkdir -p "${DATA}"
DATA="$(cd "${DATA}" && pwd)"

VISION_STEM="clipseg_vision_rd16_224"
if [[ "${IMAGE_BATCH}" -gt 1 ]]; then
  VISION_STEM="${VISION_STEM}_b${IMAGE_BATCH}"
fi
FILM_ENGINE="${DATA}/clipseg_film_rd16_224_b${CLIPSEG_FILM_BATCH}_p${CLIPSEG_FILM_PROMPTS}.engine"
VISION_ENGINE="${DATA}/${VISION_STEM}.engine"
SAM_ENC_ENGINE="${DATA}/resnet18_image_encoder_512_dyn.engine"
SAM_DEC_ENGINE="${DATA}/mobile_sam_mask_decoder_batched_512.engine"

echo "=========================================="
echo " HOUND perception build-all"
echo " DATA=${DATA}"
echo " CLIPSeg image batch=${IMAGE_BATCH} FiLM prompts=${CLIPSEG_FILM_PROMPTS}"
echo " NanoSAM encoder 512 dyn opt/max=${ENC_OPT_BATCH}/${ENC_MAX_BATCH}"
echo " NanoSAM decoder opt=${OPT_BATCH} max=${MAX_BATCH}"
echo " USE_DLA=${USE_DLA} USE_DLA_DYNAMIC=${USE_DLA_DYNAMIC}"
echo "=========================================="

if [[ "${SKIP_CLIPSEG}" != "1" ]]; then
  echo ""
  echo ">>> [1/4] CLIPSeg vision ONNX → TensorRT FP16 (DLA 0) batch=${IMAGE_BATCH}"
  DLA_CORE=0 bash "${HOUND_SCRIPTS}/build_clipseg_vision_engine.sh" \
    "${DATA}" "CIDAS/clipseg-rd16" "224" "${IMAGE_BATCH}"
else
  echo ">>> [1/4] CLIPSeg vision SKIPPED"
fi

if [[ "${SKIP_CLIPSEG_FILM}" != "1" ]]; then
  echo ""
  echo ">>> [2/4] CLIPSeg FiLM decoder ONNX → TensorRT FP16 (DLA 1)"
  echo "    batch=${CLIPSEG_FILM_BATCH} prompts=${CLIPSEG_FILM_PROMPTS}"
  DLA_CORE=1 bash "${HOUND_SCRIPTS}/build_clipseg_film_engine.sh" \
    "${DATA}" "CIDAS/clipseg-rd16" "224" "${CLIPSEG_FILM_BATCH}" "${CLIPSEG_FILM_PROMPTS}"
else
  echo ">>> [2/4] CLIPSeg FiLM SKIPPED"
fi

if [[ "${SKIP_NANOSAM}" != "1" ]]; then
  echo ""
  echo ">>> [3/4] NanoSAM 512 pair + dynamic encoder batch ${ENC_OPT_BATCH}..${ENC_MAX_BATCH}"
  # Pair writes static 512 ONNX/engine + decoder; then patch encoder to dyn batch.
  DLA_CORE=0 bash "${HOUND_SCRIPTS}/build_nanosam_512_pair.sh" \
    "${DATA}" "${OPT_BATCH}" "${MAX_BATCH}"
  echo ">>> [3b] NanoSAM encoder dynamic batch (DLA 0)"
  DLA_CORE=0 bash "${HOUND_SCRIPTS}/build_nanosam_image_encoder_batched.sh" \
    "${DATA}" "512" "${ENC_OPT_BATCH}" "${ENC_MAX_BATCH}"
  DEC_ONNX="${DATA}/mobile_sam_mask_decoder_batched_512.onnx"
  if [[ -f "${DEC_ONNX}" ]]; then
    echo ">>> [3c] Re-bind NanoSAM decoder to DLA 1 (opt=${OPT_BATCH} max=${MAX_BATCH})"
    DLA_CORE=1 bash "${HOUND_SCRIPTS}/build_nanosam_mask_decoder_batched_engine.sh" \
      "${DEC_ONNX}" "${SAM_DEC_ENGINE}" "${OPT_BATCH}" "${MAX_BATCH}" 32 32 128
  fi
else
  echo ">>> [3/4] NanoSAM SKIPPED"
fi

if [[ "${SKIP_YOLO}" != "1" ]]; then
  echo ""
  echo ">>> [4/4] YOLO-World stash embeddings → *_stashed.pt (torch FP16 runtime)"
  echo "    (ONNX/TRT export is blocked on WorldModel adaptive_max_pool2d)"
  python3 "${HOUND_SCRIPTS}/export_yolo_world.py" \
    --model "${YOLO_MODEL}" \
    --classes "${YOLO_CLASSES[@]}" \
    --out-dir "${DATA}" \
    --format pt
else
  echo ">>> [4/4] YOLO-World SKIPPED"
fi

echo ""
echo "=========================================="
echo " Done. Artifacts under ${DATA}:"
ls -lh "${DATA}"/*.{engine,onnx,pt} 2>/dev/null || ls -lh "${DATA}" | tail -20
echo ""
echo "Point SSoT (hound_core/config/SSoT.yaml) at:"
echo "  segmentation.clipseg_vision_engine: ${VISION_ENGINE}"
echo "  segmentation.clipseg_film_engine:   ${FILM_ENGINE}"
echo "  segmentation.sam_image_encoder:     ${SAM_ENC_ENGINE}"
echo "  segmentation.sam_mask_decoder:      ${SAM_DEC_ENGINE}"
echo "  segmentation.clipseg_batch_n:       ${IMAGE_BATCH}"
echo "  segmentation.sam_image_size:        512"
echo "  yolo_world.model:                   ${DATA}/yolov8s-world_stashed.pt  # if exported"
echo ""
echo "Note: rebuild engines on this device. If TRT exposes DLA, unsupported layers"
echo "use GPU via --allowGPUFallback. Force GPU: USE_DLA=0 bash .../build_all_engines.sh"
echo ""
echo "Optional benches:"
echo "  python3 ${HOUND_SCRIPTS}/bench_clipseg.py --trt-engine ${VISION_ENGINE}"
echo "  python3 ${HOUND_SCRIPTS}/bench_yolo_world.py --models ${DATA}/yolov8s-world_stashed.pt"
echo "=========================================="
