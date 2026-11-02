#!/usr/bin/env python3
"""Dora node: YOLO-World open-vocabulary instance detector (dora only, no ROS).

Cold path: class prompts → CLIP text embeddings, computed on a background
thread (:class:`PromptBank`) and swapped in atomically. That is the same
contract as the CLIPSeg prompt path: changing prompts never stalls frames, and
every frame is detected with one consistent prompt set.

Hot path: camera frame(s) → boxes (torch FP16; ONNX/TRT export of the
Ultralytics WorldModel is blocked, see ``yolo_world/trt.py``).

Inputs:

* camera inputs listed in ``camera_inputs`` (default: ids starting with
  ``image`` or ``cam_``): self-describing ``camera_frame`` messages. A batch is
  formed when the leader camera delivers a frame (``LatestFrameSync``).
* ``prompt``: new class prompts (Arrow string array or ``{"prompts": [...]}``).

Output ``detections``: self-describing ``detections`` message
(``dora_io.encode_detections``) stamped with the source frames' stamps, so
``dora_seg_refine`` (``object_source: yolo_world``) can pair the boxes with the
frame it has already SAM-encoded. A frame that arrives before any prompt is
active still gets an (empty) detections message, so downstream masks clear.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from dora import Node

from perception_models.dora_io import (
    LatestFrameSync,
    WireError,
    decode_camera_frame,
    encode_detections,
    is_camera_input,
    parse_prompts,
)
from perception_models.prompt_bank import PromptBank
from perception_models.yolo_world.shared import YoloWorldParts, load_yolo_config, prompt_layout_from_config


class YoloWorldNode:
    def __init__(self, cfg: dict):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; yolo_world needs a GPU")
        self.cfg = cfg
        self.dora = Node()
        self.profile = bool(cfg.get("profile", False))
        self.batch_n = max(1, int(cfg.get("clipseg_batch_n", cfg.get("batch_n", 1))))

        layout = prompt_layout_from_config(cfg)
        self.parts = YoloWorldParts(
            model_name=str(cfg.get("yolo_model", cfg.get("model", "yolov8s-worldv2.pt"))),
            prompts=layout,
            conf=float(cfg.get("conf", 0.05)),
            iou=float(cfg.get("iou", 0.5)),
            device=str(cfg.get("device", "cuda:0")),
            imgsz=int(cfg.get("imgsz", 640)),
            half=bool(cfg.get("half", True)),
            text_model=str(cfg.get("text_model", "openai/clip-vit-base-patch32")),
            logger=self.log,
        )
        self.parts.load()
        self.prompts = PromptBank(
            self.parts.embed_classes,
            max_prompts=cfg.get("max_classes"),
            on_update=lambda s: self.log(f"prompts v{s.version} active: {list(s.prompts)}"),
            on_error=lambda ps, e: self.log(f"WARN prompt embedding failed for {list(ps)}: {e}"),
        )
        if layout.classes:
            self.prompts.set_blocking(layout.classes)
        self.parts.warmup(self.prompts.current())

        self.camera_inputs = [str(c) for c in (cfg.get("camera_inputs") or []) if str(c)]
        self._sync: LatestFrameSync | None = LatestFrameSync(self.camera_inputs) if self.camera_inputs else None
        self._prof: list[float] = []
        self.log(
            f"dora yolo_world up: model={self.parts.model_name} imgsz={self.parts.imgsz} "
            f"conf={self.parts.conf} classes={layout.classes} camera_inputs={self.camera_inputs or 'auto'}"
        )

    @staticmethod
    def log(msg: str) -> None:
        print(f"[dora_yolo_world] {msg}", flush=True)

    # ------------------------------------------------------------ hot path
    def _on_camera_frame(self, input_id: str, event) -> None:
        try:
            frame = decode_camera_frame(event["value"])
        except WireError as e:
            self.log(f"WARN {input_id}: {e}")
            return
        if self._sync is None:
            self._sync = LatestFrameSync([input_id])
        if input_id not in self._sync.cameras:
            if len(self._sync.cameras) >= self.batch_n:
                return
            self._sync.cameras.append(input_id)
        batch = self._sync.push(input_id, frame)
        if batch is None:
            return
        snap = self.prompts.current()  # one snapshot for the whole batch
        rgbs = [f["rgb"] for f in batch]
        t0 = time.perf_counter()
        dets = self.parts.detect_batch(rgbs, snap) if snap is not None else [None] * len(rgbs)
        if self.profile:
            self._prof.append((time.perf_counter() - t0) * 1e3)
            if len(self._prof) >= 30:
                self.log(f"[timing] yolo detect {np.mean(self._prof):.1f} ms/batch (n={len(self._prof)})")
                self._prof.clear()
        cameras = []
        for f, cam, d in zip(batch, self._sync.cameras, dets):
            h, w = f["rgb"].shape[:2]
            boxes, scores, cls = d if d is not None else (np.zeros((0, 4)), np.zeros(0), np.zeros(0))
            cameras.append(
                {
                    "name": str(f.get("name") or cam),
                    "stamp": float(f["stamp"]),
                    "height": int(h),
                    "width": int(w),
                    "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    "scores": np.asarray(scores, dtype=np.float32).reshape(-1),
                    "class_idx": np.asarray(cls, dtype=np.int32).reshape(-1),
                }
            )
        stamp = float(batch[0]["stamp"])
        self.dora.send_output(
            "detections",
            encode_detections(
                stamp=stamp,
                prompts=list(snap.prompts) if snap is not None else [],
                prompt_version=snap.version if snap is not None else 0,
                cameras=cameras,
            ),
            {"stamp": stamp, "n_det": int(sum(len(c["scores"]) for c in cameras))},
        )

    # -------------------------------------------------------------- events
    def run(self) -> None:
        try:
            for event in self.dora:
                if event is None or event.get("type") == "STOP":
                    break
                if event.get("type") != "INPUT":
                    continue
                eid = str(event.get("id"))
                if eid == "prompt":
                    try:
                        prompts = parse_prompts(event["value"])
                        if prompts and self.prompts.request(prompts):
                            self.log(f"prompt update queued: {prompts}")
                    except Exception as e:  # noqa: BLE001
                        self.log(f"WARN bad prompt update: {e}")
                elif is_camera_input(eid, self.camera_inputs):
                    self._on_camera_frame(eid, event)
        finally:
            self.prompts.close()


def main():
    import argparse

    ap = argparse.ArgumentParser(description="YOLO-World dora detector")
    ap.add_argument("--config", default=None, help="JSON/YAML node config (default: $HOUND_YOLO_CONFIG)")
    args, _ = ap.parse_known_args()
    YoloWorldNode(load_yolo_config(args.config)).run()


if __name__ == "__main__":
    main()
