"""YOLO-World TensorRT vision/detect (prompt embeddings as engine inputs).

Contract (CLIPSeg-parallel):

* **Offline:** CLIP text encode → class embedding bank ``[C, D]`` (numpy/torch).
* **Engine:** image ``[1,3,H,W]`` (+ optional embeddings input) → boxes.
* **Reload stash** when prompts change; do not rebuild engine unless shape ``C``
  or reparameterized weights change.

## Status (2026-07)

Ultralytics WorldModel ONNX export fails on ``adaptive_max_pool2d`` (unsupported
op). Until a custom ONNX/TRT path exists:

* **Ship torch FP16** via ``YoloWorldParts`` + optional ``*_stashed.pt``
  (``scripts/export_yolo_world.py --format pt``).
* Point SSoT ``yolo_world.model`` at the stashed weights for cold-start without
  re-running ``set_classes`` from scratch (still calls ``set_classes`` on load).
* Leave ``yolo_world_engine`` empty.

This module remains the loader stub for when an engine lands under
``perception_models/data/``.
"""

from __future__ import annotations

from pathlib import Path


class YoloWorldTrtUnavailable(RuntimeError):
    """Raised when TRT path is requested but not supported yet."""


def load_yolo_world_engine(engine_path: str | Path):
    """Load TensorRT engine for YOLO-World detect path.

    Raises ``FileNotFoundError`` / ``YoloWorldTrtUnavailable`` until export works.
    """
    path = Path(engine_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO-World TRT engine not found: {path}. "
            "Use YoloWorldParts (torch FP16) or "
            "scripts/export_yolo_world.py --format pt for stashed weights. "
            "ONNX/TRT blocked: WorldModel adaptive_max_pool2d."
        )
    raise YoloWorldTrtUnavailable(
        "YOLO-World TRT runtime not wired yet (ONNX export blocked on "
        f"adaptive_max_pool2d). Engine present but unused: {path}. "
        "Fall back to YoloWorldParts torch FP16."
    )
