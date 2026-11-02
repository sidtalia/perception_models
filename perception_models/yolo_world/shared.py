"""YOLO-World prompt stash + Ultralytics detect (CLIPSeg-shaped API).

Hot path = image → boxes. Prompt / CLIP text encode happens only when the
class list changes (``set_prompts``), same idea as CLIPSeg prompt tensors.
"""

from __future__ import annotations

import json
import os
import threading
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
    """YOLO-World detector with prompt embeddings decoupled from the model.

    The text side (CLIP ViT-B/32, the encoder YOLO-World-v2 was trained with)
    runs in :meth:`embed_classes`, which is safe to call from a background
    thread: it only touches its own text model and CUDA stream. The result is
    a ``[1, C, 512]`` tensor that :meth:`detect_batch` installs on the model
    right before each forward, so one frame always sees one consistent prompt
    set and prompt changes never block the frame loop. That is the same
    contract as CLIPSeg's ``cond`` tensors (see ``PromptBank``).

    ``set_prompts`` / ``detect_boxes`` remain as the simple synchronous API
    (used by ``scripts/bench_yolo_world.py``).
    """

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        prompts: YoloWorldPromptLayout | None = None,
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str = "cuda:0",
        imgsz: int = 640,
        half: bool = True,
        text_model: str = "openai/clip-vit-base-patch32",
        logger: Callable[[str], None] | None = None,
    ):
        self.model_name = model_name
        self.prompts = prompts or YoloWorldPromptLayout()
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self.text_model_name = text_model
        self._log = logger or (lambda _m: None)
        self._model = None
        self._world = None  # the ultralytics WorldModel the predictor actually runs
        self._text = None
        self._text_lock = threading.Lock()
        self._text_stream = None
        self._installed: object | None = None
        self._stashed_classes: list[str] | None = None

    # ------------------------------------------------------------ loading
    def load(self) -> None:
        """Load detector weights and the CLIP text encoder (idempotent)."""
        if self._model is not None:
            return
        # Robot runtime: never pip-install at import, never phone home (usage analytics,
        # hub checks). Weights must be a local file; set these env vars yourself to override.
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        os.environ.setdefault("YOLO_OFFLINE", "true")
        import torch
        from ultralytics import YOLO

        if not Path(self.model_name).is_file():
            raise FileNotFoundError(
                f"YOLO-World weights not found: {self.model_name} "
                "(download e.g. yolov8l-worldv2.pt from github.com/ultralytics/assets/releases)"
            )

        self._log(f"Loading YOLO-World '{self.model_name}' ...")
        self._model = YOLO(self.model_name)
        # The first predict builds the predictor (AutoBackend, fuse, FP16 cast);
        # afterwards the WorldModel it runs is the one we install embeddings on.
        self._model.predict(
            np.zeros((self.imgsz, self.imgsz, 3), np.uint8),
            imgsz=self.imgsz, half=self.half, device=self.device, verbose=False,
        )  # fmt: skip
        self._world = _find_world_model(self._model)
        self._text_stream = torch.cuda.Stream() if str(self.device).startswith("cuda") else None
        self._load_text()
        self._log(f"YOLO-World ready (half={self.half}, imgsz={self.imgsz}); text: {self.text_model_name}")

    def _load_text(self) -> None:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        with self._text_lock:
            if self._text is None:
                tok = CLIPTokenizer.from_pretrained(self.text_model_name)
                enc = CLIPTextModelWithProjection.from_pretrained(self.text_model_name).to(self.device).eval()
                self._text = (tok, enc)

    # ------------------------------------------------------------- prompts
    def embed_classes(self, classes: Sequence[str]):
        """Class prompts → L2-normalised CLIP text embeddings ``[1, C, D]`` (thread-safe)."""
        import torch

        names = [str(c).strip() for c in classes if str(c).strip()]
        if not names:
            raise ValueError("no class prompts")
        if self._text is None:
            self._load_text()
        tok, enc = self._text
        ctx = torch.cuda.stream(self._text_stream) if self._text_stream is not None else _nullcontext()
        with self._text_lock, torch.inference_mode(), ctx:
            batch = tok(names, padding=True, return_tensors="pt").to(self.device)
            feats = enc(**batch).text_embeds
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            if self._text_stream is not None:
                self._text_stream.synchronize()
        return feats.reshape(1, len(names), -1).float()

    def _install(self, txt_feats) -> None:
        """Make the next forward use ``txt_feats`` ``[1, C, D]`` (frame thread only)."""
        if self._installed is txt_feats:
            return
        world = self._world
        world.txt_feats = txt_feats
        world.model[-1].nc = int(txt_feats.shape[1])
        names = {i: str(i) for i in range(int(txt_feats.shape[1]))}
        world.names = names
        pred = getattr(self._model, "predictor", None)
        if pred is not None and getattr(pred, "model", None) is not None:
            pred.model.names = names
        self._installed = txt_feats

    def set_prompts(self, classes: Sequence[str]) -> None:
        """Synchronously embed and install ``classes`` (simple API; not for the frame loop)."""
        self.load()
        names = [str(c).strip() for c in classes if str(c).strip()] or ["person"]
        self.prompts = YoloWorldPromptLayout(classes=names, ensure_person=self.prompts.ensure_person)
        self._cond = self.embed_classes(self.prompts.classes)
        self._install(self._cond)
        self._stashed_classes = list(self.prompts.classes)
        self._log(f"stashed class embeddings: {self._stashed_classes}")

    # ------------------------------------------------------------ hot path
    def warmup(self, snapshot=None) -> None:
        self.load()
        cond = getattr(snapshot, "cond", None)
        if cond is None:
            cond = self.embed_classes(self.prompts.classes or ["object"])
        rgb = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(3):
            self._predict([rgb], cond)

    def _predict(self, rgbs: list[np.ndarray], txt_feats, conf: float | None = None):
        self._install(txt_feats)
        return self._model.predict(
            source=[np.ascontiguousarray(r[:, :, ::-1]) for r in rgbs],  # ultralytics expects BGR arrays
            conf=float(self.conf if conf is None else conf),
            iou=self.iou,
            imgsz=self.imgsz,
            half=self.half,
            device=self.device,
            verbose=False,
        )

    def detect_batch(self, rgbs: list[np.ndarray], snapshot, conf: float | None = None):
        """Detect on N RGB images with one prompt snapshot.

        Args:
            rgbs: HxWx3 uint8 RGB images.
            snapshot: object with ``cond`` (``[1, C, D]`` from :meth:`embed_classes``).
            conf: confidence threshold (default: ``self.conf``).

        Returns:
            Per image ``(boxes [K,4] xyxy float32, scores [K] float32, class_idx [K] int32)``,
            best first.
        """
        results = self._predict(rgbs, snapshot.cond, conf)
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32)))
                continue
            xyxy = r.boxes.xyxy.float().cpu().numpy()
            sc = r.boxes.conf.float().cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(np.int32)
            order = np.argsort(-sc)
            out.append((xyxy[order].astype(np.float32), sc[order].astype(np.float32), cl[order]))
        return out

    def detect_boxes(self, rgb: np.ndarray, *, conf: float | None = None) -> list[dict[str, Any]]:
        """Per-frame detect with the stashed prompts. rgb HxWx3 uint8 RGB → list of box dicts.

        Each dict: ``xyxy`` (4 ints), ``conf`` float, ``cls`` int, ``name`` str.
        """
        if self._stashed_classes is None:
            self.set_prompts(self.prompts.classes)
        h, w = rgb.shape[:2]
        (xyxy, confs, clss), = self.detect_batch([rgb], _Snap(self._cond), conf)
        names = self.prompts.classes
        out: list[dict[str, Any]] = []
        for box, c, k in zip(xyxy, confs, clss):
            x0, y0, x1, y1 = [int(round(float(v))) for v in box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            ki = int(k)
            out.append({"xyxy": (x0, y0, x1, y1), "conf": float(c), "cls": ki,
                        "name": names[ki] if 0 <= ki < len(names) else str(ki)})  # fmt: skip
        return out


@dataclass(frozen=True)
class _Snap:
    cond: Any


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _find_world_model(yolo: Any) -> Any:
    """The ``WorldModel`` instance the predictor runs (AutoBackend may wrap it)."""
    candidates = []
    pred = getattr(yolo, "predictor", None)
    if pred is not None:
        backend = getattr(pred, "model", None)
        candidates += [getattr(backend, "model", None), backend]
    candidates.append(getattr(yolo, "model", None))
    for m in candidates:
        if m is not None and hasattr(m, "txt_feats") and hasattr(m, "model"):
            return m
    raise RuntimeError("could not find the YOLO-World WorldModel (is this a *-world*.pt checkpoint?)")
