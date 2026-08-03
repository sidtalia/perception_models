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

Under `data/` (rebuild per Jetson — do not copy `.engine` across boxes):

```bash
bash scripts/build_all_engines.sh
```

Upstream NanoSAM tooling: `nanosam-src/` (`pip install -e`).
