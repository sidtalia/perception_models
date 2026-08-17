#!/usr/bin/env bash
# Build all HOUND perception model artifacts on *this* GPU box (Orin NX / Thor).
#
# TensorRT engines must be rebuilt per JetPack/TRT/GPU — do not copy .engine
# files between machines. ONNX / stashed .pt can be reused as inputs.
#
# Inside mushr_jazzy (or Thor container with trtexec + CUDA):
#   bash /root/colcon_ws/src/perception_models/scripts/build_all_engines.sh
#
# Env overrides:
#   DATA=...          output dir (default: perception_models/data)
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
OPT_BATCH="${OPT_BATCH:-16}"
MAX_BATCH="${MAX_BATCH:-32}"
USE_DLA="${USE_DLA:-1}"
USE_DLA_DYNAMIC="${USE_DLA_DYNAMIC:-1}"
SKIP_CLIPSEG="${SKIP_CLIPSEG:-0}"
SKIP_CLIPSEG_FILM="${SKIP_CLIPSEG_FILM:-0}"
SKIP_NANOSAM="${SKIP_NANOSAM:-0}"
SKIP_YOLO="${SKIP_YOLO:-0}"
CLIPSEG_FILM_BATCH="${CLIPSEG_FILM_BATCH:-3}"
CLIPSEG_FILM_PROMPTS="${CLIPSEG_FILM_PROMPTS:-9}"
YOLO_MODEL="${YOLO_MODEL:-yolov8s-world.pt}"
# shellcheck disable=SC2206
YOLO_CLASSES=(${YOLO_CLASSES:-person chair door table bag})

export USE_DLA USE_DLA_DYNAMIC

mkdir -p "${DATA}"
DATA="$(cd "${DATA}" && pwd)"

echo "=========================================="
echo " HOUND perception build-all"
echo " DATA=${DATA}"
echo " NanoSAM TRT profile: opt=${OPT_BATCH} max=${MAX_BATCH}"
echo " USE_DLA=${USE_DLA} USE_DLA_DYNAMIC=${USE_DLA_DYNAMIC}"
echo "=========================================="

if [[ "${SKIP_CLIPSEG}" != "1" ]]; then
  echo ""
  echo ">>> [1/4] CLIPSeg vision ONNX → TensorRT FP16 (DLA 0)"
  DLA_CORE=0 bash "${HOUND_SCRIPTS}/build_clipseg_vision_engine.sh" "${DATA}"
else
  echo ">>> [1/4] CLIPSeg vision SKIPPED"
fi

if [[ "${SKIP_CLIPSEG_FILM}" != "1" ]]; then
  echo ""
  echo ">>> [2/4] CLIPSeg FiLM decoder ONNX → TensorRT FP16 (DLA 1)"
  DLA_CORE=1 bash "${HOUND_SCRIPTS}/build_clipseg_film_engine.sh" \
    "${DATA}" "CIDAS/clipseg-rd16" "224" "${CLIPSEG_FILM_BATCH}" "${CLIPSEG_FILM_PROMPTS}"
  # Also build batch=1 for single-cam.
  if [[ "${CLIPSEG_FILM_BATCH}" != "1" ]]; then
    DLA_CORE=1 bash "${HOUND_SCRIPTS}/build_clipseg_film_engine.sh" \
      "${DATA}" "CIDAS/clipseg-rd16" "224" "1" "${CLIPSEG_FILM_PROMPTS}"
  fi
else
  echo ">>> [2/4] CLIPSeg FiLM SKIPPED"
fi

if [[ "${SKIP_NANOSAM}" != "1" ]]; then
  echo ""
  echo ">>> [3/4] NanoSAM 512 encoder (DLA 0) + batched mask decoder (DLA 1)"
  # Pair script builds encoder then calls decoder script; set encoder core here,
  # then rebuild decoder alone on DLA 1 (pair would inherit DLA 0 for both).
  DLA_CORE=0 bash "${HOUND_SCRIPTS}/build_nanosam_512_pair.sh" \
    "${DATA}" "${OPT_BATCH}" "${MAX_BATCH}"
  DEC_ENGINE="${DATA}/mobile_sam_mask_decoder_batched_512.engine"
  DEC_ONNX="${DATA}/mobile_sam_mask_decoder_batched_512.onnx"
  if [[ -f "${DEC_ONNX}" ]]; then
    echo ">>> [3b] Re-bind NanoSAM decoder to DLA 1"
    DLA_CORE=1 bash "${HOUND_SCRIPTS}/build_nanosam_mask_decoder_batched_engine.sh" \
      "${DEC_ONNX}" "${DEC_ENGINE}" "${OPT_BATCH}" "${MAX_BATCH}" 32 32 128
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
echo "  segmentation.clipseg_vision_engine: ${DATA}/clipseg_vision_rd16_224.engine"
echo "  segmentation.clipseg_film_engine:   ${DATA}/clipseg_film_rd16_224_b1_p${CLIPSEG_FILM_PROMPTS}.engine"
echo "  segmentation.sam_image_encoder:     ${DATA}/resnet18_image_encoder_512.engine"
echo "  segmentation.sam_mask_decoder:      ${DATA}/mobile_sam_mask_decoder_batched_512.engine"
echo "  yolo_world.model:                   ${DATA}/yolov8s-world_stashed.pt  # if exported"
echo ""
echo "Note: rebuild engines on this device. If TRT exposes DLA, unsupported layers"
echo "use GPU via --allowGPUFallback. Force GPU: USE_DLA=0 bash .../build_all_engines.sh"
echo ""
echo "Optional benches:"
echo "  python3 ${HOUND_SCRIPTS}/bench_clipseg.py --trt-engine ${DATA}/clipseg_vision_rd16_224.engine"
echo "  python3 ${HOUND_SCRIPTS}/bench_yolo_world.py --models ${DATA}/yolov8s-world_stashed.pt"
echo "=========================================="
