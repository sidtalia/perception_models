"""Multi-cam RGB batch pack/unpack for Dora (N×H×W×3 uint8 contiguous)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pyarrow as pa


def camera_ns_from_color_topic(topic: str) -> str:
    """``/camera_front/color/image_raw`` → ``/camera_front``."""
    t = str(topic or "").strip()
    if not t:
        return ""
    if not t.startswith("/"):
        t = "/" + t
    t = t.rstrip("/")
    for suffix in (
        "/color/image_raw",
        "/color/image_rect_raw",
        "/color/image_rect",
        "/image_raw",
    ):
        if t.endswith(suffix):
            return t[: -len(suffix)] or ""
    parts = [p for p in t.split("/") if p]
    if len(parts) >= 2:
        return "/" + "/".join(parts[:-2]) if len(parts) > 2 else "/" + parts[0]
    return t


def refined_trav_topic(cam_ns: str) -> str:
    """``/camera_front`` → ``/camera_front/segmentation/refined_traversability``."""
    ns = str(cam_ns or "").rstrip("/")
    if not ns:
        return "/segmentation/refined_traversability"
    if not ns.startswith("/"):
        ns = "/" + ns
    return f"{ns}/segmentation/refined_traversability"


def refined_people_mask_topic(cam_ns: str) -> str:
    ns = str(cam_ns or "").rstrip("/")
    if not ns:
        return "/segmentation/refined_people_mask"
    if not ns.startswith("/"):
        ns = "/" + ns
    return f"{ns}/segmentation/refined_people_mask"


def pack_rgb_batch(
    rgbs: Sequence[np.ndarray],
    *,
    frame_ids: Sequence[str],
    stamp_sec: Sequence[int],
    stamp_nanosec: Sequence[int],
) -> Tuple[pa.Array, Dict[str, Any]]:
    if not rgbs:
        raise ValueError("empty rgb batch")
    n = len(rgbs)
    if not (len(frame_ids) == len(stamp_sec) == len(stamp_nanosec) == n):
        raise ValueError("meta length mismatch")
    h, w = int(rgbs[0].shape[0]), int(rgbs[0].shape[1])
    stacked = np.stack(
        [np.ascontiguousarray(r, dtype=np.uint8).reshape(h, w, 3) for r in rgbs],
        axis=0,
    )
    meta: Dict[str, Any] = {
        "n_cams": n,
        "height": h,
        "width": w,
        "frame_ids": list(frame_ids),
        # Lists for multi-cam; scalars for Dora matching / single-cam refine.
        "stamp_sec_list": [int(x) for x in stamp_sec],
        "stamp_nanosec_list": [int(x) for x in stamp_nanosec],
        "stamp_sec": int(stamp_sec[0]),
        "stamp_nanosec": int(stamp_nanosec[0]),
        "stamp_sec0": int(stamp_sec[0]),
        "stamp_nanosec0": int(stamp_nanosec[0]),
        "frame_id": str(frame_ids[0]),
    }
    return pa.array(stacked.reshape(-1), type=pa.uint8()), meta


def unpack_rgb_batch(buf, meta: Dict[str, Any]) -> List[np.ndarray]:
    n = int(meta.get("n_cams", 1))
    h = int(meta["height"])
    w = int(meta["width"])
    if hasattr(buf, "to_numpy"):
        flat = np.asarray(buf.to_numpy(), dtype=np.uint8)
    elif hasattr(buf, "values"):
        flat = np.asarray(buf.values.to_numpy(), dtype=np.uint8)
    else:
        flat = np.asarray(buf, dtype=np.uint8)
    arr = flat.reshape(n, h, w, 3)
    return [arr[i].copy() for i in range(n)]
