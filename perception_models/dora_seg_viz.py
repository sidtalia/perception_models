#!/usr/bin/env python3
"""Dora → Rerun visualization for segmentation overlays (no ROS).

Keeps up with the pipeline by JPEG-compressing frames and never blocking
refine: dataflow inputs use ``queue_size: 1`` (latest-only).
"""

from __future__ import annotations

import os

import numpy as np
import rerun as rr

from dora import Node


def _to_rgb(buf, meta) -> np.ndarray:
    h = int(meta["height"])
    w = int(meta["width"])
    if hasattr(buf, "to_numpy"):
        flat = np.asarray(buf.to_numpy(), dtype=np.uint8)
    elif hasattr(buf, "values"):
        flat = np.asarray(buf.values.to_numpy(), dtype=np.uint8)
    else:
        flat = np.asarray(buf, dtype=np.uint8)
    return flat.reshape(h, w, 3)


def main():
    app = os.environ.get("RERUN_APP_ID", "hound_seg")
    spawn = os.environ.get("RERUN_SPAWN", "1").strip() not in ("0", "false", "False")
    jpeg_q = int(os.environ.get("RERUN_JPEG_QUALITY", "70") or "70")
    rr.init(app, spawn=spawn)
    print(
        f"[dora_seg_viz] rerun app={app} spawn={spawn} jpeg_q={jpeg_q}",
        flush=True,
    )

    dora = Node()
    for event in dora:
        if event is None or event.get("type") == "STOP":
            break
        if event.get("type") != "INPUT":
            continue
        eid = event.get("id")
        meta = dict(event.get("metadata") or {})
        try:
            img = _to_rgb(event["value"], meta)
        except Exception as exc:  # noqa: BLE001
            print(f"[dora_seg_viz] skip {eid}: {exc}", flush=True)
            continue
        e2e = meta.get("e2e_ms")
        entity = f"seg/{eid}"
        image = rr.Image(img)
        if jpeg_q > 0:
            try:
                image = image.compress(jpeg_quality=jpeg_q)
            except Exception:  # noqa: BLE001
                pass
        rr.log(entity, image)
        if e2e is not None:
            rr.log("seg/e2e_ms", rr.Scalar(float(e2e)))


if __name__ == "__main__":
    main()
