#!/usr/bin/env bash
# Build NanoSAM ResNet18 image-encoder TRT engine with dynamic batch (1..MAX).
#
#   bash scripts/build_nanosam_image_encoder_batched.sh \
#     [data_dir] [size=512] [opt_batch=3] [max_batch=4]
#
# Writes:
#   resnet18_image_encoder_${SIZE}_dyn.onnx
#   resnet18_image_encoder_${SIZE}_dyn.engine
#
# Input  image            [N,3,SIZE,SIZE]
# Output image_embeddings [N,256,SIZE/16,SIZE/16]
set -euo pipefail

HOUND_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
DATA="${1:-${HOUND_SCRIPTS}/../data}"
SIZE="${2:-512}"
OPT_BATCH="${3:-3}"
MAX_BATCH="${4:-4}"
EMB=$((SIZE / 16))

SRC_ONNX="${DATA}/resnet18_image_encoder_${SIZE}.onnx"
if [[ ! -f "${SRC_ONNX}" ]]; then
  SRC_ONNX="${DATA}/resnet18_image_encoder.onnx"
fi
if [[ ! -f "${SRC_ONNX}" ]]; then
  echo "Missing encoder ONNX (expected ${DATA}/resnet18_image_encoder_${SIZE}.onnx)" >&2
  exit 1
fi

DST_ONNX="${DATA}/resnet18_image_encoder_${SIZE}_dyn.onnx"
DST_ENGINE="${DATA}/resnet18_image_encoder_${SIZE}_dyn.engine"

echo "=== 1) Patch ONNX batch dim → dynamic (from ${SRC_ONNX}) ==="
python3 - <<EOF
import onnx
from onnx import shape_inference

src, dst = "${SRC_ONNX}", "${DST_ONNX}"
size, emb = int("${SIZE}"), int("${EMB}")
m = onnx.load(src)
for inp in m.graph.input:
    if inp.name != "image":
        continue
    dims = inp.type.tensor_type.shape.dim
    dims[0].ClearField("dim_value")
    dims[0].dim_param = "batch"
    for i, v in ((1, 3), (2, size), (3, size)):
        dims[i].ClearField("dim_param")
        dims[i].dim_value = int(v)
for out in m.graph.output:
    if out.name != "image_embeddings":
        continue
    dims = out.type.tensor_type.shape.dim
    dims[0].ClearField("dim_value")
    dims[0].dim_param = "batch"
    for i, v in ((1, 256), (2, emb), (3, emb)):
        dims[i].ClearField("dim_param")
        dims[i].dim_value = int(v)
del m.graph.value_info[:]
m = shape_inference.infer_shapes(m)
onnx.save(m, dst)
print("wrote", dst)
EOF

echo "=== 2) Build TensorRT engine (FP16, batch 1..${MAX_BATCH}, opt=${OPT_BATCH}) ==="
TRTEXEC="$(command -v trtexec || true)"
if [[ -z "${TRTEXEC}" && -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [[ -z "${TRTEXEC}" ]]; then
  echo "trtexec not found" >&2
  exit 1
fi

"${TRTEXEC}" \
  --onnx="${DST_ONNX}" \
  --saveEngine="${DST_ENGINE}" \
  --fp16 \
  --minShapes="image:1x3x${SIZE}x${SIZE}" \
  --optShapes="image:${OPT_BATCH}x3x${SIZE}x${SIZE}" \
  --maxShapes="image:${MAX_BATCH}x3x${SIZE}x${SIZE}" \
  --skipInference

echo "=== Done ==="
echo "Engine: ${DST_ENGINE}"
echo "Point SSoT segmentation.sam_image_encoder at this path for batched multicam encode."
