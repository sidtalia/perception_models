# YOLO-World + CLIPSeg (+ shared NanoSAM) — design note
#
# Mental model: same as CLIPSeg dora nodes. The *difference* is what each head
# is good at predicting, not how caching works.

## Parallel to CLIPSeg

| | CLIPSeg | YOLO-World |
|--|---------|------------|
| Job | nav semantics (trav / nontrav / …) | instance objects (+ person) |
| Stash (cold) | CLIP text → FiLM prompt tensors | CLIP text → class embeddings (`set_classes`) |
| Hot path | vision encode (TRT) → FiLM maps | vision+detect (torch FP16) → boxes |
| Refine | NanoSAM nav decode (trav/nontrav CCs) | NanoSAM manip decode (YOLO boxes) |
| Consumer | nvblox / nav | spencer_port persons + object masks |

```
/camera/color/image_raw
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌───────────────────┐        ┌────────────────────┐
│ dora_clipseg_*    │        │ dora_yolo_world    │
│ stash prompts     │        │ stash class embeds │
│ TRT vision + FiLM │        │ torch FP16 detect  │
│ (no people if YOLO│        │                    │
│  owns people)     │        │                    │
└─────────┬─────────┘        └─────────┬──────────┘
          │ trav/nontrav CC boxes        │ /yolo_world/detections
          └──────────────┬───────────────┘
                         ▼
              NanoSAM in seg_refine:
                1× encode → decode_nav + decode_manip
                (separate box budgets; people never starve trav)
```

When `yolo_world.enabled`, launch strips CLIPSeg `people` prompts and sets
`yolo_owns_people` on seg_refine. NanoSAM then:

* **nav decode** — CLIPSeg trav/nontrav only → `/segmentation/refined_traversability`
* **manip decode** — YOLO person/object boxes → `/segmentation/refined_people_mask`

Budgets: `sam_max_boxes` (nav), `sam_max_boxes_manip` (manip).

## Packages

* **Models:** `perception_models/yolo_world/` (`YoloWorldParts`, stash + detect)
* **Dora node:** `perception_models/perception_models/dora_yolo_world.py` + `scripts/dora_yolo_world`
* **Dataflow:** `perception_models/dora/yolo_world_dataflow.yml`
* **Config:** SSoT `yolo_world:` → launch writes `HOUND_YOLO_CONFIG`
* **spencer:** `--vision yolo_dets` subscribes to `/yolo_world/detections` (or
  `--vision yolo_world` for in-process detect)

## Enable

```yaml
# SSoT.yaml
yolo_world:
  enabled: true
  classes: [person, chair, door]
  # Prefer stashed weights after export_yolo_world.py --format pt:
  # model: "/root/colcon_ws/src/perception_models/data/yolov8s-world_stashed.pt"
```

## Bench / export

```bash
# FPS sweep (cached prompts), torch FP16:
python3 /root/colcon_ws/src/perception_models/scripts/bench_yolo_world.py

# Stash embeddings → *_stashed.pt (supported). ONNX/TRT usually fails.
python3 /root/colcon_ws/src/perception_models/scripts/export_yolo_world.py \
  --classes person chair door --format pt
```

## TRT status

Blocked: Ultralytics WorldModel ONNX export hits unsupported `adaptive_max_pool2d`.
Production path = torch FP16 + stashed `.pt`. Loader stub: `yolo_world/trt.py`.

## Status

* [x] Prompt stash + torch FP16 detect (`YoloWorldParts`)
* [x] Dora node + launch + SSoT
* [x] Dual-decode NanoSAM (nav trav + manip people; separate budgets)
* [x] CLIPSeg skips people when YOLO owns people
* [x] spencer `--vision yolo_dets` → `/yolo_world/detections`
* [ ] TRT engine load (`yolo_world/trt.py`) — blocked on ONNX
* [x] dora-io branch: ROS-free node (`camera_frame` in, `detections` out); prompts via
      `PromptBank` (background CLIP text embed, atomic swap per frame)
* [x] `seg_refine` `object_source: yolo_world`: SAM encode on frame arrival ∥ detection,
      stamp-matched detections → one batched decode (CLIPSeg not loaded)
* [ ] Nav (CLIPSeg) + manip (YOLO) dual decode on the dora-io branch: refine runs one
      source at a time today

### dora-io notes

* Text embeddings: `transformers` CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), identical
  to ultralytics' `clip:ViT-B/32` (cos 1.0); the `clip` git package is not needed.
* `YoloWorldParts.load()` sets `YOLO_AUTOINSTALL=false` and `YOLO_OFFLINE=true` (no runtime
  pip installs, no analytics) and requires `yolo_model` to be a local file.
* Measured on the PRAXIS YCB scene (RTX PRO 6000, 640x480, `yolov8l-worldv2`, FP16): ~38 ms
  per frame; prompts "mustard bottle" / "soup can" / "mug" all detected (0.34 / 0.76 / 0.86).
