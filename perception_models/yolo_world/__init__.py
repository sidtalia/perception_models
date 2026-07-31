"""YOLO-World backends: prompt-then-detect (stashed text embeddings + vision).

Mirrors ``perception_models.clipseg``:

* **CLIPSeg:** stash CLIP text for FiLM; TRT the vision tower per frame.
* **YOLO-World:** stash CLIP text via ``set_classes`` / offline vocab; TRT
  (or torch) the detect path per frame. Text encoder is *not* on the hot path.

Dora/ROS nodes live in ``hound_core``; this package owns model + engines.
"""

from __future__ import annotations

from perception_models.yolo_world.shared import (
    YoloWorldParts,
    YoloWorldPromptLayout,
    load_yolo_config,
    prompt_layout_from_config,
)

__all__ = [
    "YoloWorldParts",
    "YoloWorldPromptLayout",
    "load_yolo_config",
    "prompt_layout_from_config",
]
