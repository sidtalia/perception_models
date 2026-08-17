#!/usr/bin/env bash
# Build a matched NanoSAM pair at input size 512 (features 32x32):
#   resnet18_image_encoder_512.{onnx,engine}
#   mobile_sam_mask_decoder_batched_512.{onnx,engine}
#
# Keeps existing *_1024 / un-suffixed 1024 artifacts untouched so you can A/B
# via SSoT:
#   sam_image_size: 512
#   sam_image_encoder: .../resnet18_image_encoder_512.engine
#   sam_mask_decoder: .../mobile_sam_mask_decoder_batched_512.engine
#
# Run inside mushr_jazzy (GPU + trtexec).
set -euo pipefail

DATA="${1:-/root/colcon_ws/src/perception_models/data}"
OPT_BATCH="${2:-6}"
MAX_BATCH="${3:-16}"
SIZE=512
EMB=$((SIZE / 16))          # 32
MASK_IN=$((4 * EMB))        # 128

ENC_ONNX_SRC="${DATA}/resnet18_image_encoder.onnx"
ENC_ONNX="${DATA}/resnet18_image_encoder_${SIZE}.onnx"
ENC_ENGINE="${DATA}/resnet18_image_encoder_${SIZE}.engine"
DEC_ONNX="${DATA}/mobile_sam_mask_decoder_batched_${SIZE}.onnx"
DEC_ENGINE="${DATA}/mobile_sam_mask_decoder_batched_${SIZE}.engine"
CKPT="${NANOSAM_CKPT:-/root/colcon_ws/src/nanosam-src/assets/mobile_sam.pt}"
HOUND_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo "=== 1) Encoder ONNX @ ${SIZE} (from ${ENC_ONNX_SRC}) ==="
python3 - <<EOF
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
import torch
import torch.nn.functional as F

src = "${ENC_ONNX_SRC}"
dst = "${ENC_ONNX}"
size, emb = ${SIZE}, ${EMB}
m = onnx.load(src)
new_inits = []
for init in m.graph.initializer:
    if init.name == "pos_embedding":
        arr = np.array(numpy_helper.to_array(init), copy=True)
        t32 = F.interpolate(
            torch.from_numpy(arr), size=(emb, emb), mode="bilinear", align_corners=False
        )
        new_inits.append(numpy_helper.from_array(t32.numpy(), name="pos_embedding"))
        print(f"pos_embedding {arr.shape} -> {tuple(t32.shape)}")
    else:
        new_inits.append(init)
del m.graph.initializer[:]
m.graph.initializer.extend(new_inits)
for inp in m.graph.input:
    if inp.name == "image":
        for i, d in enumerate(inp.type.tensor_type.shape.dim):
            if i >= 2:
                d.ClearField("dim_param")
                d.dim_value = size
for out in m.graph.output:
    if out.name == "image_embeddings":
        dims = out.type.tensor_type.shape.dim
        vals = [1, 256, emb, emb]
        for i, v in enumerate(vals):
            dims[i].ClearField("dim_param")
            dims[i].dim_value = v
del m.graph.value_info[:]
m = shape_inference.infer_shapes(m)
onnx.save(m, dst)
print("wrote", dst)
EOF

echo "=== 2) Encoder TRT engine ==="
# shellcheck source=trt_common.sh
source "${HOUND_SCRIPTS}/trt_common.sh"
resolve_trtexec
configure_trt_dla_flags static
"${TRTEXEC}" \
  --onnx="${ENC_ONNX}" \
  --saveEngine="${ENC_ENGINE}" \
  --fp16 \
  "${TRT_DLA_FLAGS[@]}" \
  --skipInference
echo "wrote ${ENC_ENGINE}"

echo "=== 3) Batched decoder ONNX @ ${SIZE} (embed ${EMB}) ==="
python3 "${HOUND_SCRIPTS}/export_nanosam_mask_decoder_batched_onnx.py" \
  --checkpoint "${CKPT}" \
  --output "${DEC_ONNX}" \
  --image-size "${SIZE}"

echo "=== 4) Decoder TRT engine (opt=${OPT_BATCH} max=${MAX_BATCH}) ==="
bash "${HOUND_SCRIPTS}/build_nanosam_mask_decoder_batched_engine.sh" \
  "${DEC_ONNX}" "${DEC_ENGINE}" "${OPT_BATCH}" "${MAX_BATCH}" "${EMB}" "${EMB}" "${MASK_IN}"

echo "=== Done ==="
echo "A/B in SSoT.yaml:"
echo "  sam_image_size: ${SIZE}"
echo "  sam_image_encoder: \"${ENC_ENGINE}\""
echo "  sam_mask_decoder: \"${DEC_ENGINE}\""
echo "Swap back to 1024 engines + sam_image_size: 1024 for the baseline."
