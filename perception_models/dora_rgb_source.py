#!/usr/bin/env python3
"""Dora-only RGB source — rosbag → packed Dora ``rgb`` (no ROS pub/sub).

Stamps ``t_src_perf = time.perf_counter()`` at send for e2e latency.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from dora import Node

from perception_models.clipseg.shared import load_seg_config
from perception_models.multicam_rgb_batch import pack_rgb_batch


def _msg_to_rgb(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = str(getattr(msg, "encoding", "bgr8") or "bgr8").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc.startswith("rgb8"):
        return np.ascontiguousarray(buf.reshape(h, w, 3))
    if enc.startswith("bgr8"):
        return np.ascontiguousarray(buf.reshape(h, w, 3)[:, :, ::-1])
    if enc in ("mono8", "8uc1"):
        g = buf.reshape(h, w)
        return np.ascontiguousarray(np.stack([g, g, g], axis=-1))
    raise ValueError(f"unsupported encoding {enc!r}")


def _log(msg: str) -> None:
    print(f"[dora_rgb_source] {msg}", flush=True)


def _send_batch(dora, rgbs, frame_ids, secs, nsecs):
    t_src = time.perf_counter()
    arr, meta = pack_rgb_batch(
        rgbs, frame_ids=frame_ids, stamp_sec=secs, stamp_nanosec=nsecs
    )
    meta["source"] = "dora_rgb_source"
    meta["t_src_perf"] = float(t_src)
    dora.send_output("rgb", arr, meta)


def _iter_bag_images(bag_path: Path, topic: str, loop: bool):
    from rosbags.highlevel import AnyReader

    while True:
        with AnyReader([bag_path]) as reader:
            conns = [c for c in reader.connections if c.topic == topic]
            if not conns:
                topics = sorted({c.topic for c in reader.connections})
                raise RuntimeError(
                    f"topic {topic!r} not in bag; sample={topics[:16]}"
                )
            for conn, bag_t, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                yield bag_t, msg
        if not loop:
            return
        _log("bag loop restart")


def main():
    cfg = load_seg_config()
    dora = Node()

    bag_path = Path(str(cfg.get("bag_path", "") or "")).expanduser()
    if not bag_path.is_file():
        raise FileNotFoundError(
            f"dora_rgb_source: bag_path required (isolation, no ROS). got={bag_path}"
        )

    topic = str(cfg.get("bag_color_topic") or "/camera_left/color/image_raw")
    camera_names = [str(n) for n in (cfg.get("camera_names") or []) if str(n)]
    fanout_identical = bool(cfg.get("fanout_identical", False))
    if fanout_identical and len(camera_names) < 2:
        camera_names = ["front", "left", "right"]
    n_out = len(camera_names) if fanout_identical else 1
    if not camera_names:
        camera_names = ["cam0"]

    input_hz = float(cfg.get("input_hz", 0) or 0)
    min_dt = (1.0 / input_hz) if input_hz > 0 else 0.0
    loop = bool(cfg.get("bag_loop", True))
    max_frames = int(cfg.get("bag_max_frames", 0) or 0)

    _log(
        f"bag={bag_path.name} topic={topic} n_out={n_out} "
        f"fanout_identical={fanout_identical} input_hz={input_hz or 'uncapped'} "
        f"loop={loop}"
    )

    msg_iter = _iter_bag_images(bag_path, topic, loop)
    n_sent = 0
    last_pub_t = 0.0

    for event in dora:
        if event is None or event.get("type") == "STOP":
            break
        if event.get("type") != "INPUT" or event.get("id") != "tick":
            continue

        now = time.perf_counter()
        if min_dt > 0 and (now - last_pub_t) < min_dt:
            continue

        try:
            _bag_t, msg = next(msg_iter)
        except StopIteration:
            _log(f"bag end after {n_sent} frames")
            break
        last_pub_t = now

        rgb = _msg_to_rgb(msg)
        stamp_sec = int(msg.header.stamp.sec)
        stamp_nanosec = int(msg.header.stamp.nanosec)

        if fanout_identical:
            rgbs = [rgb] * n_out
            frame_ids = [f"{n}_optical" for n in camera_names[:n_out]]
            secs = [stamp_sec] * n_out
            nsecs = [stamp_nanosec] * n_out
        else:
            rgbs = [rgb]
            frame_ids = [str(msg.header.frame_id or "cam0")]
            secs = [stamp_sec]
            nsecs = [stamp_nanosec]

        _send_batch(dora, rgbs, frame_ids, secs, nsecs)
        n_sent += 1
        if max_frames > 0 and n_sent >= max_frames:
            _log(f"bag_max_frames={max_frames} reached; stopping")
            break


if __name__ == "__main__":
    main()
