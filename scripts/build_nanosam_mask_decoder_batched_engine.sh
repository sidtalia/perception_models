#!/usr/bin/env bash
# Build a TensorRT mask-decoder engine that accepts a batch of box prompts.
# Run inside the GPU container (mushr_jazzy / Jetson) where trtexec + CUDA work.
#
# Usage:
#   bash scripts/build_nanosam_mask_decoder_batched_engine.sh \
#     [/path/to/mobile_sam_mask_decoder_batched.onnx] \
#     [/path/to/output.engine] \
#     [opt_batch] \
#     [max_batch] \
#     [embed_h=64] \
#     [embed_w=64] \
#     [mask_input_side=256]
#
# For a 512 encoder (embed 32): pass 32 32 128 as the last three args.
set -euo pipefail

ONNX="${1:-/root/colcon_ws/src/perception_models/data/mobile_sam_mask_decoder_batched.onnx}"
ENGINE="${2:-/root/colcon_ws/src/perception_models/data/mobile_sam_mask_decoder_batched.engine}"
OPT_BATCH="${3:-6}"
MAX_BATCH="${4:-16}"
EMB_H="${5:-64}"
EMB_W="${6:-64}"
MASK_IN="${7:-256}"

# MobileSAM feature map channels
EMB_C=256

if [[ ! -f "$ONNX" ]]; then
  echo "ONNX not found: $ONNX"
  echo "Export first:"
  echo "  python3 /root/colcon_ws/src/hound_core/scripts/export_nanosam_mask_decoder_batched_onnx.py \\"
  echo "    --output $ONNX"
  exit 1
fi

if ! [[ "$OPT_BATCH" =~ ^[0-9]+$ && "$MAX_BATCH" =~ ^[0-9]+$ ]]; then
  echo "opt_batch and max_batch must be positive integers (got opt=$OPT_BATCH max=$MAX_BATCH)"
  exit 1
fi
if (( OPT_BATCH < 1 || MAX_BATCH < 1 )); then
  echo "opt_batch and max_batch must be >= 1"
  exit 1
fi
if (( OPT_BATCH > MAX_BATCH )); then
  echo "opt_batch ($OPT_BATCH) must be <= max_batch ($MAX_BATCH)"
  exit 1
fi

MIN_SHAPES="image_embeddings:1x${EMB_C}x${EMB_H}x${EMB_W},point_coords:1x2x2,point_labels:1x2,mask_input:1x1x${MASK_IN}x${MASK_IN},has_mask_input:1"
OPT_SHAPES="image_embeddings:${OPT_BATCH}x${EMB_C}x${EMB_H}x${EMB_W},point_coords:${OPT_BATCH}x2x2,point_labels:${OPT_BATCH}x2,mask_input:${OPT_BATCH}x1x${MASK_IN}x${MASK_IN},has_mask_input:${OPT_BATCH}"
MAX_SHAPES="image_embeddings:${MAX_BATCH}x${EMB_C}x${EMB_H}x${EMB_W},point_coords:${MAX_BATCH}x2x2,point_labels:${MAX_BATCH}x2,mask_input:${MAX_BATCH}x1x${MASK_IN}x${MASK_IN},has_mask_input:${MAX_BATCH}"

echo "Building $ENGINE from $ONNX (min=1 opt=$OPT_BATCH max=$MAX_BATCH embed=${EMB_H}x${EMB_W} mask_in=${MASK_IN}) ..."
# --skipInference: avoid trtexec's post-build smoke test fighting profile shapes;
# the engine file is what we need. Re-run without it later to benchmark if desired.
trtexec \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --skipInference \
  --minShapes="$MIN_SHAPES" \
  --optShapes="$OPT_SHAPES" \
  --maxShapes="$MAX_SHAPES"

echo "Done. Point SSoT segmentation.sam_mask_decoder at:"
echo "  $ENGINE"
echo "And keep segmentation.sam_max_boxes <= $MAX_BATCH (opt profile tuned for $OPT_BATCH)."
