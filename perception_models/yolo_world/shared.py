"""YOLO-World prompt stash + Ultralytics detect (CLIPSeg-shaped API).

Hot path = image → boxes. Prompt / CLIP text encode happens only when the
class list changes (``set_prompts``), same idea as CLIPSeg prompt tensors.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml


@dataclass
class YoloWorldPromptLayout:
    """Open-vocab class names (offline vocabulary).

    Parallel to CLIPSeg ``PromptLayout``: navigation groups there, instance
    class names here. Changing ``classes`` triggers a re-stash, not a retrain.
    """

    classes: list[str] = field(default_factory=lambda: ["person"])
    ensure_person: bool = True

    def __post_init__(self) -> None:
        self.classes = [str(c).strip() for c in self.classes if str(c).strip()]
        if not self.classes:
            self.classes = ["person"]
        if self.ensure_person and "person" not in self.classes:
            self.classes = ["person"] + list(self.classes)

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def prompt_layout_from_config(cfg: dict[str, Any]) -> YoloWorldPromptLayout:
    yw = cfg.get("yolo_world") if isinstance(cfg.get("yolo_world"), dict) else None
    if yw is None and ("classes" in cfg or "prompts" in cfg or "model" in cfg):
        yw = cfg
    yw = yw or {}
    classes = yw.get("classes") or yw.get("prompts") or ["person"]
    return YoloWorldPromptLayout(
        classes=list(classes),
        ensure_person=bool(yw.get("ensure_person", True)),
    )


def load_yolo_config(path: str | None = None) -> dict[str, Any]:
    """Load YOLO-World JSON/YAML from ``HOUND_YOLO_CONFIG`` (launch-written)."""
    cfg_path = path or os.environ.get("HOUND_YOLO_CONFIG", "")
    if not cfg_path:
        # Fallbacks for ad-hoc runs / spencer configs.
        for key in ("HOUND_SEG_CONFIG", "SPENCER_PORT_CONFIG"):
            alt = os.environ.get(key, "").strip()
            if alt and Path(alt).is_file():
                cfg_path = alt
                break
    if not cfg_path:
        raise RuntimeError("HOUND_YOLO_CONFIG is not set")
    p = Path(cfg_path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    if isinstance(raw.get("yolo_world"), dict):
        # Merge: yolo block wins; keep color_topic from parent if needed.
        merged = dict(raw)
        merged.update(raw["yolo_world"])
        return merged
    return raw


class YoloWorldParts:
    """Load YOLO-World once; stash text embeddings via ``set_classes``.

    ``detect_boxes()`` is the per-frame path (torch today; TRT engine later).
    """

    def __init__(
        self,
        model_name: str = "yolov8s-world.pt",
        prompts: YoloWorldPromptLayout | None = None,
        *,
        conf: float = 0.25,
        device: str = "cuda:0",
        imgsz: int = 640,
        half: bool = True,
        logger: Callable[[str], None] | None = None,
    ):
        self.model_name = model_name
        self.prompts = prompts or YoloWorldPromptLayout()
        self.conf = float(conf)
        self.device = device
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self._log = logger or (lambda _m: None)
        self._model = None
        self._stashed_classes: list[str] | None = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._log(f"Loading YOLO-World '{self.model_name}' ...")
        self._model = YOLO(self.model_name)
        self.set_prompts(self.prompts.classes)
        if self.half and str(self.device).startswith("cuda"):
            try:
                if hasattr(self._model, "model") and self._model.model is not None:
                    self._model.model.half()
                    self._log("YOLO-World weights: FP16")
            except Exception as exc:  # noqa: BLE001
                self._log(f"FP16 cast skipped: {exc}")
        self._log(
            f"YOLO-World ready; stashed {len(self.prompts.classes)} class embeddings"
        )

    def set_prompts(self, classes: Sequence[str]) -> None:
        """Re-stash offline vocabulary (CLIP text encode once). Not per-frame."""
        self._ensure()
        names = [str(c).strip() for c in classes if str(c).strip()]
        if not names:
            names = ["person"]
        ensure = self.prompts.ensure_person
        self.prompts = YoloWorldPromptLayout(classes=names, ensure_person=ensure)
        if hasattr(self._model, "set_classes"):
            self._model.set_classes(self.prompts.classes)
        self._stashed_classes = list(self.prompts.classes)
        self._log(f"stashed class embeddings: {self._stashed_classes}")

    def warmup(self) -> None:
        self._ensure()
        rgb = np.zeros((min(64, self.imgsz), min(64, self.imgsz), 3), dtype=np.uint8)
        _ = self.detect_boxes(rgb)

    def detect_boxes(
        self,
        rgb: np.ndarray,
        *,
        conf: float | None = None,
    ) -> list[dict[str, Any]]:
        """Per-frame detect. rgb HxWx3 uint8 RGB → list of box dicts.

        Each dict: ``xyxy`` (4 ints), ``conf`` float, ``cls`` int, ``name`` str.
        """
        self._ensure()
        h, w = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        results = self._model.predict(
            source=bgr,
            conf=float(self.conf if conf is None else conf),
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        out: list[dict[str, Any]] = []
        if not results:
            return out
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return out
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = (
            r0.boxes.cls.cpu().numpy().astype(np.int32)
            if r0.boxes.cls is not None
            else np.zeros((len(xyxy),), dtype=np.int32)
        )
        names = self.prompts.classes
        for box, c, k in zip(xyxy, confs, clss):
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            ki = int(k)
            name = names[ki] if 0 <= ki < len(names) else str(ki)
            out.append(
                {
                    "xyxy": (x0, y0, x1, y1),
                    "conf": float(c),
                    "cls": ki,
                    "name": name,
                }
            )
        return out
