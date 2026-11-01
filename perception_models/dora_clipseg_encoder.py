#!/usr/bin/env python3
"""CLIPSeg vision encoder (dora only, no ROS).

Inputs (any combination):

* ``rgb``: packed multi-cam batch from ``dora_rgb_source`` (isolation / bags).
* camera inputs listed in ``camera_inputs`` (default: every input whose id starts
  with ``image`` or ``cam_``): self-describing ``camera_frame`` messages, e.g.
  straight from a PRAXIS robot node. The node keeps the newest frame per camera
  and encodes when the *leader* (first camera in ``camera_inputs``) delivers a
  frame and every camera has one. This is the latest-frame sync the ROS edge
  used, without ROS.

Outputs: ``features`` (CLIPSeg hidden states, Arrow f16) and the forwarded
``rgb`` batch. Both share the same stamp metadata so ``dora_seg_refine`` can
pair them. ``stamp_list`` carries the original float stamps for exact depth
matching downstream.
"""

from __future__ import annotations

import os
import time

import torch

from dora import Node

from perception_models.clipseg.shared import (
    ClipSegParts,
    load_seg_config,
    prompt_layout_from_config,
)
from perception_models.clipseg.trt import load_clipseg_vision_engine, trt_encode_hs
from perception_models.cuda_ipc import pack_arrow_f16
from perception_models.dora_io import (
    LatestFrameSync,
    WireError,
    decode_camera_frame,
    is_camera_input,
    split_stamp,
)
from perception_models.multicam_rgb_batch import pack_rgb_batch, unpack_rgb_batch


class ClipSegEncoderNode:
    """Batched CLIPSeg vision TRT encode; forwards rgb + features."""

    def __init__(self, cfg: dict):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; clipseg encoder needs a GPU")
        torch.set_float32_matmul_precision("high")

        self.cfg = cfg
        self.dora = Node()
        self.profile = os.environ.get("PROFILE", "0") == "1"

        layout = prompt_layout_from_config(cfg)
        trt_engine_path = str(cfg.get("clipseg_vision_engine", "") or "").strip()
        if not trt_engine_path:
            raise RuntimeError("clipseg_vision_engine required (TRT-only encoder)")
        self.batch_n = max(1, int(cfg.get("clipseg_batch_n", 1)))

        self.parts = ClipSegParts(
            str(cfg.get("model_name", "CIDAS/clipseg-rd16")),
            layout,
            input_res=int(cfg.get("input_res", 224)),
            compile_mode="none",
            device="cuda",
            logger=self._log,
        )

        self._log(f"Loading CLIPSeg vision TensorRT engine: {trt_engine_path}")
        self.trt_enc = load_clipseg_vision_engine(trt_engine_path)
        dummy = torch.zeros(self.batch_n, 3, self.parts.res, self.parts.res, device="cuda")
        for _ in range(3):
            _ = trt_encode_hs(self.trt_enc, dummy)
        torch.cuda.synchronize()
        self._log(f"CLIPSeg TRT vision warmup complete (batch_n={self.batch_n}).")

        self.camera_inputs = [str(c) for c in (cfg.get("camera_inputs") or []) if str(c)]
        self._sync: LatestFrameSync | None = (
            LatestFrameSync(self.camera_inputs) if self.camera_inputs else None
        )
        if self._sync is not None and len(self.camera_inputs) != self.batch_n:
            raise RuntimeError(
                f"{len(self.camera_inputs)} camera_inputs but clipseg_batch_n={self.batch_n}; "
                "engines are built for a fixed image batch"
            )
        self._encode_ok = 0
        self._log(
            f"dora clipseg_encoder up: batch_n={self.batch_n} "
            f"camera_inputs={self.camera_inputs or 'auto (image*/cam_*)'}"
        )

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[dora_clipseg_encoder] {msg}", flush=True)

    @staticmethod
    def _warn(msg: str) -> None:
        print(f"[dora_clipseg_encoder] WARN {msg}", flush=True)

    # ----------------------------------------------------------- encode core
    def _encode_and_send(
        self, rgbs, frame_ids, stamp_sec, stamp_nanosec, *, src_meta=None, stamps=None
    ) -> None:
        shapes = {tuple(r.shape[:2]) for r in rgbs}
        if len(shapes) != 1:
            self._warn(f"skip encode: mismatched RGB shapes {[r.shape[:2] for r in rgbs]}")
            return
        if len(rgbs) != self.batch_n:
            self._warn(f"skip encode: got {len(rgbs)} images, engine batch is {self.batch_n}")
            return

        h, w = rgbs[0].shape[:2]
        pix = self.parts.preprocess_rgb_batch(rgbs)
        if self.profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        hs = trt_encode_hs(self.trt_enc, pix)
        if self.profile:
            torch.cuda.synchronize()
            print(f"CLIPSeg encode time: {(time.perf_counter() - t0) * 1000.0} ms", flush=True)
        if hs.dim() == 3:
            hs = hs.unsqueeze(0)

        n_cams = int(hs.shape[0])
        meta = {
            "n_cams": n_cams,
            "stamp_sec_list": list(stamp_sec),
            "stamp_nanosec_list": list(stamp_nanosec),
            "frame_ids": list(frame_ids),
            "stamp_sec": int(stamp_sec[0]),
            "stamp_nanosec": int(stamp_nanosec[0]),
            "frame_id": str(frame_ids[0]),
            "height": int(h),
            "width": int(w),
            "n_layers": int(hs.shape[1]),
            "seq_len": int(hs.shape[2]),
            "hidden": int(hs.shape[3]),
            "encode_backend": "trt",
            "feature_batch": n_cams,
            "feature_mode": "arrow_f16",
        }
        if stamps is not None:
            meta["stamp_list"] = [float(s) for s in stamps]
        if src_meta and src_meta.get("t_src_perf") is not None:
            meta["t_src_perf"] = float(src_meta["t_src_perf"])

        rgb_arr, rgb_meta = pack_rgb_batch(
            rgbs, frame_ids=frame_ids, stamp_sec=stamp_sec, stamp_nanosec=stamp_nanosec
        )
        self.dora.send_output("rgb", rgb_arr, {**meta, **rgb_meta})
        self.dora.send_output("features", pack_arrow_f16(hs), meta)
        self._encode_ok += 1

    # ------------------------------------------------------------ inputs
    def _on_camera_frame(self, input_id: str, event) -> None:
        try:
            frame = decode_camera_frame(event["value"])
        except WireError as e:
            self._warn(f"{input_id}: {e}")
            return
        frame["_t_rx"] = time.perf_counter()
        if self._sync is None:
            # Auto mode: discover cameras in arrival order; first seen leads.
            self._sync = LatestFrameSync([input_id])
            self._log(f"auto camera_inputs: leader={input_id}")
        if input_id not in self._sync.cameras:
            if len(self._sync.cameras) >= self.batch_n:
                return
            self._sync.cameras.append(input_id)
            self._log(f"auto camera_inputs: + {input_id}")
        batch = self._sync.push(input_id, frame)
        if batch is None:
            return
        stamps = [f["stamp"] for f in batch]
        split = [split_stamp(s) for s in stamps]
        self._encode_and_send(
            [f["rgb"] for f in batch],
            [str(f.get("name") or cam) for f, cam in zip(batch, self._sync.cameras)],
            [s for s, _ in split],
            [ns for _, ns in split],
            src_meta={"t_src_perf": batch[0]["_t_rx"]},
            stamps=stamps,
        )

    def _on_rgb(self, event) -> None:
        """Packed ``rgb`` (isolation / bag) → unpack → encode."""
        meta = dict(event.get("metadata") or {})
        try:
            rgbs = unpack_rgb_batch(event["value"], meta)
        except Exception as e:  # noqa: BLE001
            self._warn(f"unpack rgb failed: {e}")
            return
        n = len(rgbs)
        frame_ids = [str(x) for x in (meta.get("frame_ids") or [])]
        if len(frame_ids) < n:
            frame_ids = frame_ids + [f"cam{i}" for i in range(len(frame_ids), n)]
        stamp_sec = [int(x) for x in (meta.get("stamp_sec_list") or [])]
        stamp_nanosec = [int(x) for x in (meta.get("stamp_nanosec_list") or [])]
        if len(stamp_sec) < n:
            stamp_sec = [int(meta.get("stamp_sec", 0) or 0)] * n
            stamp_nanosec = [int(meta.get("stamp_nanosec", 0) or 0)] * n
        self._encode_and_send(
            rgbs, frame_ids[:n], stamp_sec[:n], stamp_nanosec[:n], src_meta=meta
        )

    def run(self) -> None:
        for event in self.dora:
            if event is None or event.get("type") == "STOP":
                break
            if event.get("type") != "INPUT":
                continue
            eid = str(event.get("id"))
            if eid == "rgb":
                self._on_rgb(event)
            elif is_camera_input(eid, self.camera_inputs):
                self._on_camera_frame(eid, event)


def main():
    ClipSegEncoderNode(load_seg_config()).run()


if __name__ == "__main__":
    main()
