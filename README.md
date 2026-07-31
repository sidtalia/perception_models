# perception_models

CLIPSeg + NanoSAM **inference backends** and ONNX/TensorRT artifacts for HOUND.
YOLO-World backends (prompt stash + detect; TRT TBD) live under ``yolo_world/``.

Dora/ROS nodes stay in `hound_core`; this package owns model code, rebuild
scripts, and engines.

## Layout

```
perception_models/
  perception_models/
    clipseg/           # torch encode/FiLM + optional TRT vision
    yolo_world/        # stash class embeds + detect (TRT stub)
    nanosam_trt/       # NanoSamEncoder / NanoSamDecoder (uses pip nanosam)
    seg_ops.py         # probs→labels, boxes, overlay helpers
  scripts/             # export / trtexec / bench
  data/                # *.onnx / *.engine (built on this Orin)
```

Design: `hound_core/dora/SEGMENTATION.md` (CLIPSeg), `hound_core/dora/YOLO_WORLD.md`.

## Engines (default paths)

Under `data/`:

- `clipseg_vision_rd16_224.{onnx,engine}`
- `resnet18_image_encoder_512.{onnx,engine}`
- `mobile_sam_mask_decoder_batched_512.{onnx,engine}`
- (also unsuffixed / 1024 variants for A/B)

SSoT in `hound_core` points at these absolute paths. Rebuild rather than
copying engines across boxes with different TensorRT versions.

## Rebuild

On each Jetson (Orin NX / Thor), rebuild engines — do not copy `.engine` files
across boxes:

```bash
# All of: CLIPSeg TRT + NanoSAM 512 pair + YOLO-World stashed .pt
bash /root/colcon_ws/src/perception_models/scripts/build_all_engines.sh

# Or piece-wise:
bash /root/colcon_ws/src/perception_models/scripts/build_clipseg_vision_engine.sh
bash /root/colcon_ws/src/perception_models/scripts/build_nanosam_512_pair.sh \
  /root/colcon_ws/src/perception_models/data 16 32
python3 /root/colcon_ws/src/perception_models/scripts/export_yolo_world.py --format pt
```

Upstream MobileSAM / NanoSAM tooling lives in `nanosam-src/` (`pip install -e`).
Design notes: `hound_core/dora/SEGMENTATION.md`.
