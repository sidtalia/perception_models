#!/usr/bin/env python3
"""Dora-only CLIPSeg vision encoder (isolation / no ROS).

Dora ``rgb`` (packed N×H×W×3 + meta from ``dora_rgb_source``) → TRT encode →
Dora ``features`` + forwarded ``rgb``. Forwards ``t_src_perf`` for e2e timing.
Always TensorRT batched; no torch compile path.
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
from perception_models.multicam_rgb_batch import pack_rgb_batch, unpack_rgb_batch


def _log(msg: str) -> None:
    print(f"[dora_clipseg_encoder] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[dora_clipseg_encoder] WARN {msg}", flush=True)


def main():
    cfg = load_seg_config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; clipseg encoder needs a GPU")
    torch.set_float32_matmul_precision("high")

    dora = Node()

    layout = prompt_layout_from_config(cfg)
    trt_engine_path = str(cfg.get("clipseg_vision_engine", "") or "").strip()
    if not trt_engine_path:
        raise RuntimeError(
            "clipseg_vision_engine required (TRT-only isolation encoder)"
        )
    batch_n = max(1, int(cfg.get("clipseg_batch_n", 1)))
    profile = os.environ.get("PROFILE", "0") == "1"

    parts = ClipSegParts(
        str(cfg.get("model_name", "CIDAS/clipseg-rd16")),
        layout,
        input_res=int(cfg.get("input_res", 224)),
        compile_mode="none",
        device="cuda",
        logger=_log,
    )

    _log(f"Loading CLIPSeg vision TensorRT engine: {trt_engine_path}")
    trt_enc = load_clipseg_vision_engine(trt_engine_path)
    dummy = torch.zeros(batch_n, 3, parts.res, parts.res, device="cuda")
    for _ in range(3):
        _ = trt_encode_hs(trt_enc, dummy)
    torch.cuda.synchronize()
    _log(f"CLIPSeg TRT vision warmup complete (batch_n={batch_n}).")
    _log(f"dora clipseg_encoder up: Dora rgb in batch_n={batch_n} encode=trt")

    def _encode_and_send(rgbs, frame_ids, stamp_sec, stamp_nanosec, *, src_meta=None):
        shapes = {tuple(r.shape[:2]) for r in rgbs}
        if len(shapes) != 1:
            _warn(f"skip encode: mismatched RGB shapes {[r.shape[:2] for r in rgbs]}")
            return

        h, w = rgbs[0].shape[:2]
        pix = parts.preprocess_rgb_batch(rgbs)
        if profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        hs = trt_encode_hs(trt_enc, pix)
        if profile:
            torch.cuda.synchronize()
            print(
                f"CLIPSeg encode time: {(time.perf_counter() - t0) * 1000.0} ms",
                flush=True,
            )
        if hs.dim() == 3:
            hs = hs.unsqueeze(0)

        n_cams = int(hs.shape[0])
        n_layers, seq_len, hidden = (
            int(hs.shape[1]),
            int(hs.shape[2]),
            int(hs.shape[3]),
        )

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
            "n_layers": n_layers,
            "seq_len": seq_len,
            "hidden": hidden,
            "encode_backend": "trt",
            "feature_batch": n_cams,
            "feature_mode": "arrow_f16",
        }
        if src_meta and src_meta.get("t_src_perf") is not None:
            meta["t_src_perf"] = float(src_meta["t_src_perf"])

        rgb_arr, rgb_meta = pack_rgb_batch(
            rgbs, frame_ids=frame_ids, stamp_sec=stamp_sec, stamp_nanosec=stamp_nanosec
        )
        dora.send_output("rgb", rgb_arr, {**meta, **rgb_meta})
        dora.send_output("features", pack_arrow_f16(hs), meta)

    for event in dora:
        if event is None or event.get("type") == "STOP":
            break
        if event.get("type") != "INPUT" or event.get("id") != "rgb":
            continue

        meta = dict(event.get("metadata") or {})
        try:
            rgbs = unpack_rgb_batch(event["value"], meta)
        except Exception as e:  # noqa: BLE001
            _warn(f"unpack rgb failed: {e}")
            continue
        n = len(rgbs)
        frame_ids = [str(x) for x in (meta.get("frame_ids") or [])]
        if len(frame_ids) < n:
            frame_ids = frame_ids + [f"cam{i}" for i in range(len(frame_ids), n)]
        stamp_sec = [int(x) for x in (meta.get("stamp_sec_list") or [])]
        stamp_nanosec = [int(x) for x in (meta.get("stamp_nanosec_list") or [])]
        if len(stamp_sec) < n:
            s0 = int(meta.get("stamp_sec", 0) or 0)
            ns0 = int(meta.get("stamp_nanosec", 0) or 0)
            stamp_sec = [s0] * n
            stamp_nanosec = [ns0] * n
        _encode_and_send(
            rgbs,
            frame_ids[:n],
            stamp_sec[:n],
            stamp_nanosec[:n],
            src_meta=meta,
        )


if __name__ == "__main__":
    main()
