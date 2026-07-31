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
#   SKIP_CLIPSEG=1    skip CLIPSeg vision engine
#   SKIP_NANOSAM=1    skip NanoSAM 512 encoder+decoder pair
#   SKIP_YOLO=1       skip YOLO-World stashed .pt (torch path; no TRT yet)
#   YOLO_MODEL=...    ultralytics weights (default: yolov8s-world.pt)
#   YOLO_CLASSES=...  space-separated class list
set -euo pipefail

HOUND_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-${HOUND_SCRIPTS}/../data}"
OPT_BATCH="${OPT_BATCH:-16}"
MAX_BATCH="${MAX_BATCH:-32}"
SKIP_CLIPSEG="${SKIP_CLIPSEG:-0}"
SKIP_NANOSAM="${SKIP_NANOSAM:-0}"
SKIP_YOLO="${SKIP_YOLO:-0}"
YOLO_MODEL="${YOLO_MODEL:-yolov8s-world.pt}"
# shellcheck disable=SC2206
YOLO_CLASSES=(${YOLO_CLASSES:-person chair door table bag})

mkdir -p "${DATA}"
DATA="$(cd "${DATA}" && pwd)"

echo "=========================================="
echo " HOUND perception build-all"
echo " DATA=${DATA}"
echo " NanoSAM TRT profile: opt=${OPT_BATCH} max=${MAX_BATCH}"
echo "=========================================="

if [[ "${SKIP_CLIPSEG}" != "1" ]]; then
  echo ""
  echo ">>> [1/3] CLIPSeg vision ONNX → TensorRT FP16"
  bash "${HOUND_SCRIPTS}/build_clipseg_vision_engine.sh" "${DATA}"
else
  echo ">>> [1/3] CLIPSeg SKIPPED"
fi

if [[ "${SKIP_NANOSAM}" != "1" ]]; then
  echo ""
  echo ">>> [2/3] NanoSAM 512 encoder + batched mask decoder"
  bash "${HOUND_SCRIPTS}/build_nanosam_512_pair.sh" \
    "${DATA}" "${OPT_BATCH}" "${MAX_BATCH}"
else
  echo ">>> [2/3] NanoSAM SKIPPED"
fi

if [[ "${SKIP_YOLO}" != "1" ]]; then
  echo ""
  echo ">>> [3/3] YOLO-World stash embeddings → *_stashed.pt (torch FP16 runtime)"
  echo "    (ONNX/TRT export is blocked on WorldModel adaptive_max_pool2d)"
  python3 "${HOUND_SCRIPTS}/export_yolo_world.py" \
    --model "${YOLO_MODEL}" \
    --classes "${YOLO_CLASSES[@]}" \
    --out-dir "${DATA}" \
    --format pt
else
  echo ">>> [3/3] YOLO-World SKIPPED"
fi

echo ""
echo "=========================================="
echo " Done. Artifacts under ${DATA}:"
ls -lh "${DATA}"/*.{engine,onnx,pt} 2>/dev/null || ls -lh "${DATA}" | tail -20
echo ""
echo "Point SSoT (hound_core/config/SSoT.yaml) at:"
echo "  segmentation.clipseg_vision_engine: ${DATA}/clipseg_vision_rd16_224.engine"
echo "  segmentation.sam_image_encoder:     ${DATA}/resnet18_image_encoder_512.engine"
echo "  segmentation.sam_mask_decoder:      ${DATA}/mobile_sam_mask_decoder_batched_512.engine"
echo "  yolo_world.model:                   ${DATA}/yolov8s-world_stashed.pt  # if exported"
echo ""
echo "Optional benches:"
echo "  python3 ${HOUND_SCRIPTS}/bench_clipseg.py --trt-engine ${DATA}/clipseg_vision_rd16_224.engine"
echo "  python3 ${HOUND_SCRIPTS}/bench_yolo_world.py --models ${DATA}/yolov8s-world_stashed.pt"
echo "=========================================="
