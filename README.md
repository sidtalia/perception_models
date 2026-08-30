# perception_models

CLIPSeg + NanoSAM + YOLO-World **backends**, Dora nodes, and ONNX/TensorRT artifacts for HOUND.

## Layout

```
perception_models/
  perception_models/
    clipseg/ nanosam_trt/ yolo_world/   # inference backends
    dora_*.py cuda_ipc.py utils.py      # Dora ROS nodes
    seg_ops.py sam_box_merge.py
  dora/                                 # dataflows + design notes
  scripts/                              # export / trtexec / install_deps / dora wrappers
  data/                                 # *.onnx / *.engine (built on device)
  legacy/                               # old ROS clipseg/sam nodes (not launched)
```

Mission bring-up still lives in `hound_core` (SSoT + launch); it starts these
Dora graphs via `dora run`.

## Install deps

```bash
bash /root/colcon_ws/src/perception_models/scripts/install_deps.sh
```

Skips mesh-ICP / FoundationStereo / spencer_port.

## Engines

Under `data/` (rebuild per Jetson — do not copy `.engine` across boxes).

One shot for 3-cam SSoT: CLIPSeg vision+FiLM **batch 3**, **9** FiLM prompts,
NanoSAM encoder **512** dynamic batch 3, decoder opt **16** / max **32**.

```bash
# Default: FP16 + Jetson DLA (allowGPUFallback for unsupported layers)
bash scripts/build_all_engines.sh

# GPU only:
USE_DLA=0 bash scripts/build_all_engines.sh
```

Writes (then point SSoT at these):

- `clipseg_vision_rd16_224_b3.engine`
- `clipseg_film_rd16_224_b3_p9.engine`
- `resnet18_image_encoder_512_dyn.engine`
- `mobile_sam_mask_decoder_batched_512.engine`

`USE_DLA=1` (default) passes `--useDLACore` + `--allowGPUFallback` to `trtexec`
when TensorRT reports `num_DLA_cores > 0`. On JetPack 7.2 (L4T R39.2) TRT reports
0 DLA cores (NVIDIA: DLA coming in a future TRT release) — builds auto-fall back
to GPU FP16. `build_all_engines.sh` maps vision/encoder → DLA 0, FiLM/decoder → DLA 1
when DLA is available. Dynamic-shape NanoSAM: `USE_DLA_DYNAMIC=0` if needed.

Upstream NanoSAM tooling: `nanosam-src/` (`pip install -e`).
