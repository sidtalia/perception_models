"""Dora wire format helpers (no ROS).

Messages are *self-describing* length-1 Arrow struct arrays::

    {"_type": str, "_v": 1, "body": <value tree>}

Numpy arrays inside ``body`` are ``{"__nd__": True, "data": bytes, "shape": [...],
"dtype": str}``; heterogeneous lists are ``{"__list__": True, "items": {"i0": ...}}``.
This is the same layout the PRAXIS robot stack uses (``praxis.core.messages``), so
PRAXIS camera frames can be consumed directly and our mask output decodes there
without either package importing the other. The format is the contract; keep the
two implementations byte-compatible.

Camera frames (``_type == "camera_frame"``) carry at least ``name``, ``rgb``
(HxWx3 uint8) and ``stamp`` (float seconds); optionally ``depth`` (HxW float32 m),
``intrinsics`` (3x3) and ``extrinsics`` (camera pose in world, [x,y,z,qw,qx,qy,qz]).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pyarrow as pa

CODEC_VERSION = 1
_ND = "__nd__"


class WireError(ValueError):
    """Payload is not a valid self-describing message."""


def _to_tree(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {_ND: True, "data": arr.tobytes(), "shape": list(arr.shape), "dtype": arr.dtype.str}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        if not obj:
            return {"__empty__": True}
        return {str(k): _to_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return {"__list__": True, "items": {f"i{i}": _to_tree(v) for i, v in enumerate(obj)}}
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    raise WireError(f"cannot encode {type(obj).__name__}")


def _from_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get(_ND):
            shape = tuple(int(s) for s in obj["shape"])
            return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(shape).copy()
        if obj.get("__empty__"):
            return {}
        if obj.get("__list__"):
            items = obj["items"] or {}
            return [_from_tree(items[f"i{i}"]) for i in range(len(items))]
        return {k: _from_tree(v) for k, v in obj.items()}
    return obj


def encode(type_name: str, body: Any) -> pa.StructArray:
    """Encode ``body`` as a self-describing message (length-1 struct array)."""
    try:
        return pa.array([{"_type": type_name, "_v": CODEC_VERSION, "body": _to_tree(body)}])  # type: ignore[return-value]
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise WireError(f"failed to encode {type_name!r}: {exc}") from exc


def is_message(value: Any) -> bool:
    return (
        isinstance(value, pa.Array)
        and len(value) == 1
        and pa.types.is_struct(value.type)
        and value.type.get_field_index("_type") >= 0
    )


def decode(value: Any) -> tuple[str, Any]:
    """Decode into ``(type_name, body)``."""
    if not is_message(value):
        raise WireError("expected a length-1 self-describing struct array")
    row = value.to_pylist()[0]
    if row.get("_v") != CODEC_VERSION:
        raise WireError(f"unsupported codec version {row.get('_v')!r}")
    return str(row["_type"]), _from_tree(row["body"])


# --------------------------------------------------------------------------- #
# Stamps: float seconds on the wire; (sec, nanosec) kept for legacy metadata.
# --------------------------------------------------------------------------- #
def split_stamp(stamp: float) -> tuple[int, int]:
    sec = int(np.floor(stamp))
    nsec = int(round((stamp - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec, nsec = sec + 1, nsec - 1_000_000_000
    return sec, nsec


def join_stamp(sec: int, nsec: int) -> float:
    return float(sec) + 1e-9 * float(nsec)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def decode_camera_frame(value: Any) -> dict[str, Any]:
    """Self-describing ``camera_frame`` → dict with ``name``, ``rgb``, ``stamp`` (+ optional fields)."""
    type_name, body = decode(value)
    if type_name != "camera_frame":
        raise WireError(f"expected camera_frame, got {type_name!r}")
    rgb = body.get("rgb")
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise WireError(f"camera_frame {body.get('name')!r} has no HxWx3 rgb")
    body["rgb"] = np.ascontiguousarray(rgb, dtype=np.uint8)
    body["stamp"] = float(body["stamp"])
    return body


def parse_prompts(value: Any) -> list[str]:
    """Prompt update payload → list of prompt strings.

    Accepts a plain Arrow string array (``pa.array(["red mug"])``), or a
    self-describing message whose body is ``{"prompts": [...]}`` / ``{"prompt": "..."}``.
    Empty strings are dropped.
    """
    if is_message(value):
        _, body = decode(value)
        if isinstance(body, Mapping):
            raw = body.get("prompts", body.get("prompt", []))
        else:
            raw = body
    elif isinstance(value, pa.Array):
        raw = value.to_pylist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = value
    if isinstance(raw, str):
        raw = [raw]
    return [str(p).strip() for p in (raw or []) if p is not None and str(p).strip()]


# --------------------------------------------------------------------------- #
# Multi-camera sync
# --------------------------------------------------------------------------- #
class LatestFrameSync:
    """Newest frame per camera; emits a batch when the leader updates and all cameras have data."""

    def __init__(self, cameras: list[str]):
        if not cameras:
            raise ValueError("at least one camera input is required")
        self.cameras = list(cameras)
        self._latest: dict[str, dict] = {}

    def push(self, camera: str, frame: dict) -> list[dict] | None:
        if camera not in self.cameras:
            return None
        self._latest[camera] = frame
        if camera != self.cameras[0] or len(self._latest) < len(self.cameras):
            return None
        return [self._latest[c] for c in self.cameras]


def is_camera_input(input_id: str, configured: list[str]) -> bool:
    if configured:
        return input_id in configured
    return input_id.startswith(("image", "cam_"))


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def encode_seg_masks(
    *,
    stamp: float,
    prompts: list[str],
    prompt_version: int,
    cameras: list[dict[str, Any]],
) -> pa.StructArray:
    """Build the ``seg_masks`` output message.

    Args:
        stamp: stamp of the leader camera frame (seconds, as received).
        prompts: prompts the masks were decoded with.
        prompt_version: increments on every prompt update; lets consumers drop
            masks computed with a stale prompt set.
        cameras: one dict per camera::

            {"name": str, "stamp": float, "height": int, "width": int,
             "masks": uint8 [K,H,W] (0/1), "prompt_idx": int32 [K],
             "scores": float32 [K], "boxes": float32 [K,4] xyxy pixels}

            Instances are sorted by score, best first.
    """
    return encode(
        "seg_masks",
        {"stamp": float(stamp), "prompts": list(prompts), "prompt_version": int(prompt_version), "cameras": cameras},
    )


def encode_detections(
    *,
    stamp: float,
    prompts: list[str],
    prompt_version: int,
    cameras: list[dict[str, Any]],
) -> pa.StructArray:
    """Build the ``detections`` output message (open-vocabulary detector boxes).

    Args:
        stamp: stamp of the leader camera frame (seconds, as received). Consumers
            pair detections with that exact frame.
        prompts: class prompts the boxes were detected with; ``class_idx`` indexes it.
        prompt_version: increments on every prompt update.
        cameras: one dict per camera::

            {"name": str, "stamp": float, "height": int, "width": int,
             "boxes": float32 [K,4] xyxy pixels, "scores": float32 [K],
             "class_idx": int32 [K]}
    """
    return encode(
        "detections",
        {"stamp": float(stamp), "prompts": list(prompts), "prompt_version": int(prompt_version), "cameras": cameras},
    )
