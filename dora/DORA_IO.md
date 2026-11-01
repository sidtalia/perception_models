# dora-io branch: prompted segmentation without ROS

* **Input:** self-describing `camera_frame` messages (`perception_models/dora_io.py`), one dora
  input per camera, leader-triggered latest-frame sync. Packed `rgb` from `dora_rgb_source` still works.
* **Prompts:** the refine node's `prompt` input takes an Arrow string array or `{"prompts": [...]}`.
  Text embeddings run on a background thread with their own CUDA stream (`PromptBank`), and
  the frame loop swaps snapshots atomically.
* **Output:** `masks` is a `seg_masks` message holding per-camera instance masks for the prompted
  objects only, plus scores, boxes, the source frame stamp and `prompt_version`. There is no
  traversability output.
* **Unchanged:** CLIPSeg vision runs in its own process, FiLM ∥ SAM-encode run on two CUDA
  streams, then one batched SAM decode.
* **x86 / pip TensorRT:** `scripts/build_engines_x86.sh` uses the `scripts/trtexec_py` shim. The
  SAM encoder defaults to MobileSAM TinyViT (FP32; FP16 LayerNorms overflow in TRT 10.13).
  The FiLM TRT engine is self-checked against eager at startup and falls back to eager FP16
  if they disagree; TRT 10.13 miscompiles it here.
* **Offline check:** `scripts/validate_prompted_seg.py --config seg.json --image frame.npz --prompts ...`
