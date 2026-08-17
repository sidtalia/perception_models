#!/usr/bin/env python3
"""CLIPSeg vision encoder: Dora ``rgb`` and/or ROS multi-cam RGB.

Isolation: Dora ``rgb`` from ``dora_rgb_source`` (already packed) → unpack → encode.
Live: ROS ``color_topic(s)`` → queue → encode on Dora tick.
Multi-cam default is latest-frame sync (no HW genlock). Optional ApproximateTimeSync
via ``rgb_sync_mode=approx``. ROS spins on a background thread so BEST_EFFORT
ingress is not dropped while TRT encode / Dora send blocks.

Both paths share ``_encode_and_send`` → Dora ``features`` + forwarded ``rgb``.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np
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


def _color_topics_from_cfg(cfg: dict) -> list[str]:
    topics = [str(t) for t in (cfg.get("color_topics") or []) if str(t)]
    if topics:
        return topics
    one = str(cfg.get("color_topic", "") or "").strip()
    return [one] if one else []


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
            raise RuntimeError(
                "clipseg_vision_engine required (TRT-only isolation encoder)"
            )
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
        dummy = torch.zeros(
            self.batch_n, 3, self.parts.res, self.parts.res, device="cuda"
        )
        for _ in range(3):
            _ = trt_encode_hs(self.trt_enc, dummy)
        torch.cuda.synchronize()
        self._log(f"CLIPSeg TRT vision warmup complete (batch_n={self.batch_n}).")

        # Live ROS edge (ApproxSync) vs isolation Dora rgb. Not mutually
        # exclusive in the event loop, but ros_io prefers ROS and skips Dora rgb.
        self.color_topics = _color_topics_from_cfg(cfg)
        self.ros_io = bool(cfg.get("ros_io", False)) or bool(
            cfg.get("use_ros_edge", False)
        )
        self._ros = None
        self._bridge = None
        self._sync = None
        self._ros_subs: list = []
        self._sync_mode = str(cfg.get("rgb_sync_mode", "latest") or "latest").lower()
        self._latest_msgs: list = []
        self._latest_dirty: list[bool] = []
        self._latest_lock = threading.Lock()
        # Latest synced batch only (drop-oldest). Encode runs on Dora thread.
        self._pending: queue.Queue = queue.Queue(maxsize=1)
        self._spin_running = False
        self._spin_thread: threading.Thread | None = None
        # Sync diagnostics: per-cam arrivals vs sync successes.
        self._sync_diag = bool(cfg.get("rgb_sync_diag", True))
        self._sync_diag_every_s = float(cfg.get("rgb_sync_diag_every_s", 5.0) or 5.0)
        self._cam_rx: list[int] = []
        self._sync_ok = 0
        self._sync_dropped = 0  # overwritten before encode
        self._encode_ok = 0
        self._sync_stamp_span_ms: list[float] = []
        self._sync_diag_t0 = time.perf_counter()
        if self.ros_io:
            self._setup_ros_subscribers()

        self._log(
            f"dora clipseg_encoder up: encode=trt batch_n={self.batch_n} "
            f"ros_io={self.ros_io} sync_mode={self._sync_mode} "
            f"topics={self.color_topics or '-'}"
        )

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[dora_clipseg_encoder] {msg}", flush=True)

    @staticmethod
    def _warn(msg: str) -> None:
        print(f"[dora_clipseg_encoder] WARN {msg}", flush=True)

    def _setup_ros_subscribers(self) -> None:
        """ROS cam edge → pending queue (latest-frame or ApproximateTimeSync)."""
        import rclpy
        from cv_bridge import CvBridge
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        from rclpy.node import Node as RosNode
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        if not self.color_topics:
            raise RuntimeError("ros_io requires color_topic or color_topics")

        if not rclpy.ok():
            rclpy.init(args=None)
        self._ros = RosNode("dora_clipseg_encoder")
        self._bridge = CvBridge()
        slop = float(self.cfg.get("rgb_sync_slop_s", 0.05) or 0.05)
        # RELIABLE: large RGB was dropping under BEST_EFFORT (pub=10, ingress~7).
        # Shallow depth still limits encoder backlog; drop_old handles surplus.
        rgb_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(2, int(self.cfg.get("rgb_qos_depth", 2) or 2)),
        )

        if len(self.color_topics) == 1:
            sub = self._ros.create_subscription(
                Image,
                self.color_topics[0],
                self._on_rgb_ros_single,
                rgb_qos,
            )
            self._ros_subs.append(sub)
            self._log(f"ROS rgb sub: {self.color_topics[0]}")
            return

        n = len(self.color_topics)
        self._cam_rx = [0] * n

        # Default for multi-cam without HW genlock: latest-frame (not ApproxSync).
        # Three independent composite nodes stamp with now() → ~50–80 ms spans,
        # so ApproximateTimeSynchronizer systematically under-fires.
        if self._sync_mode != "approx":
            self._latest_msgs = [None] * n
            self._latest_dirty = [False] * n
            for i, topic in enumerate(self.color_topics):
                sub = self._ros.create_subscription(
                    Image,
                    topic,
                    lambda msg, idx=i: self._on_latest_cam(idx, msg),
                    rgb_qos,
                )
                self._ros_subs.append(sub)
            self._log(
                f"ROS latest-frame sync n={n} leader={self.color_topics[0]} "
                f"qos=RELIABLE/{rgb_qos.depth} topics={self.color_topics} "
                f"spin=bg_thread sync_diag={'on' if self._sync_diag else 'off'}"
            )
            return

        filts = [
            Subscriber(self._ros, Image, t, qos_profile=rgb_qos)
            for t in self.color_topics
        ]
        self._ros_subs.extend(filts)
        if self._sync_diag:
            for i, f in enumerate(filts):
                f.registerCallback(lambda msg, idx=i: self._on_cam_rx(idx, msg))
        sync_q = max(10, int(self.cfg.get("rgb_sync_queue_size", 30) or 30))
        self._sync = ApproximateTimeSynchronizer(
            filts, queue_size=sync_q, slop=slop
        )
        self._sync.registerCallback(self._on_rgb_ros)
        self._log(
            f"ROS ApproxSync slop={slop}s queue={sync_q} qos_depth={rgb_qos.depth} "
            f"topics={self.color_topics} spin=bg_thread "
            f"sync_diag={'on' if self._sync_diag else 'off'}"
        )

    def _on_latest_cam(self, idx: int, msg) -> None:
        """Keep newest frame per cam; emit on leader (cam0) when all have a frame.

        Waiting for every cam to refresh caps rate at min(ingress). Leader-trigger
        uses the latest side views (may be ~1 frame old) so sync tracks front.
        """
        if 0 <= idx < len(self._cam_rx):
            self._cam_rx[idx] += 1
        emit = None
        with self._latest_lock:
            self._latest_msgs[idx] = msg
            self._latest_dirty[idx] = True
            have_all = self._latest_msgs and all(
                m is not None for m in self._latest_msgs
            )
            # Leader = first color_topics entry (camera_front in SSoT).
            if have_all and idx == 0:
                emit = tuple(self._latest_msgs)
                self._latest_dirty = [False] * len(self._latest_dirty)
        if emit is not None:
            self._on_rgb_ros(*emit, color_topics=list(self.color_topics))
        else:
            self._maybe_log_sync_diag()

    def _on_cam_rx(self, idx: int, _msg) -> None:
        """Per-cam arrival counter (ApproxSync path)."""
        if 0 <= idx < len(self._cam_rx):
            self._cam_rx[idx] += 1
        self._maybe_log_sync_diag()

    def _stamp_s(self, msg) -> float:
        return float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)

    def _maybe_log_sync_diag(self) -> None:
        if not self._sync_diag:
            return
        now = time.perf_counter()
        dt = now - self._sync_diag_t0
        if dt < self._sync_diag_every_s:
            return
        cam_hz = [c / dt for c in self._cam_rx]
        sync_hz = self._sync_ok / dt
        enc_hz = self._encode_ok / dt
        drop_hz = self._sync_dropped / dt
        spans = self._sync_stamp_span_ms
        span_txt = (
            f"stamp_span_ms med={sorted(spans)[len(spans) // 2]:.1f} "
            f"max={max(spans):.1f}"
            if spans
            else "stamp_span_ms=n/a"
        )
        bits = []
        for i, hz in enumerate(cam_hz):
            label = f"cam{i}"
            if i < len(self.color_topics):
                for p in self.color_topics[i].split("/"):
                    if p.startswith("camera"):
                        label = p
                        break
            bits.append(f"{label}={hz:.1f}Hz")
        slop = float(self.cfg.get("rgb_sync_slop_s", 0.05) or 0.05)
        self._log(
            f"[sync_diag] mode={self._sync_mode} ingress={' '.join(bits)} | "
            f"sync={sync_hz:.1f}Hz encode={enc_hz:.1f}Hz "
            f"drop_old={drop_hz:.1f}Hz | {span_txt} | "
            f"slop={slop:.3f}s (window={dt:.1f}s)"
        )
        self._cam_rx = [0] * len(self._cam_rx)
        self._sync_ok = 0
        self._sync_dropped = 0
        self._encode_ok = 0
        self._sync_stamp_span_ms = []
        self._sync_diag_t0 = now

    def _enqueue_batch(self, item: tuple) -> None:
        """Keep only the latest synced batch (drop-oldest if encode is behind)."""
        if self._pending.full():
            try:
                self._pending.get_nowait()
                self._sync_dropped += 1
            except queue.Empty:
                pass
        try:
            self._pending.put_nowait(item)
        except queue.Full:
            self._sync_dropped += 1

    def _on_rgb_ros_single(self, msg) -> None:
        self._on_rgb_ros(msg, color_topics=[self.color_topics[0]])

    def _on_rgb_ros(self, *msgs, color_topics: list[str] | None = None) -> None:
        """Synced ROS Image(s) → enqueue raw msgs (no cv_bridge; keeps spin/GIL free)."""
        if not msgs:
            return
        if self._sync_diag and len(msgs) > 1:
            stamps = [self._stamp_s(m) for m in msgs]
            self._sync_stamp_span_ms.append((max(stamps) - min(stamps)) * 1000.0)
            self._sync_ok += 1
            self._maybe_log_sync_diag()
        t_src = time.perf_counter()
        topics = color_topics if color_topics is not None else list(self.color_topics)
        # Copy message references only — convert on the Dora/encode thread.
        self._enqueue_batch((tuple(msgs), topics, t_src))

    def _drain_pending_encode(self) -> None:
        try:
            msgs, topics, t_src = self._pending.get_nowait()
        except queue.Empty:
            return
        if self._bridge is None:
            return
        try:
            rgbs = [
                np.ascontiguousarray(
                    self._bridge.imgmsg_to_cv2(m, desired_encoding="rgb8")
                )
                for m in msgs
            ]
        except Exception as e:  # noqa: BLE001
            self._warn(f"cv_bridge conversion failed: {e}")
            return
        frame_ids = [str(m.header.frame_id or "") for m in msgs]
        stamp_sec = [int(m.header.stamp.sec) for m in msgs]
        stamp_nanosec = [int(m.header.stamp.nanosec) for m in msgs]
        self._encode_and_send(
            rgbs,
            frame_ids,
            stamp_sec,
            stamp_nanosec,
            src_meta={"t_src_perf": t_src, "color_topics": topics},
        )
        self._encode_ok += 1

    def _ros_spin_loop(self) -> None:
        import rclpy

        while self._spin_running and self._ros is not None and rclpy.ok():
            try:
                rclpy.spin_once(self._ros, timeout_sec=0.005)
            except Exception as e:  # noqa: BLE001
                self._warn(f"ros spin: {e}")
                time.sleep(0.01)

    def _encode_and_send(
        self, rgbs, frame_ids, stamp_sec, stamp_nanosec, *, src_meta=None
    ) -> None:
        shapes = {tuple(r.shape[:2]) for r in rgbs}
        if len(shapes) != 1:
            self._warn(
                f"skip encode: mismatched RGB shapes {[r.shape[:2] for r in rgbs]}"
            )
            return

        h, w = rgbs[0].shape[:2]
        pix = self.parts.preprocess_rgb_batch(rgbs)
        if self.profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        hs = trt_encode_hs(self.trt_enc, pix)
        if self.profile:
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
        # Per-cam color topic names for refine ROS egress topic derivation.
        topics = []
        if src_meta and src_meta.get("color_topics"):
            topics = [str(t) for t in src_meta["color_topics"]]
        elif self.color_topics and len(self.color_topics) == n_cams:
            topics = list(self.color_topics)
        if topics:
            meta["color_topics"] = topics

        # Downstream refine still takes Dora rgb+features (one pack on the way out).
        rgb_arr, rgb_meta = pack_rgb_batch(
            rgbs,
            frame_ids=frame_ids,
            stamp_sec=stamp_sec,
            stamp_nanosec=stamp_nanosec,
        )
        self.dora.send_output("rgb", rgb_arr, {**meta, **rgb_meta})
        self.dora.send_output("features", pack_arrow_f16(hs), meta)

    def _on_rgb(self, event) -> None:
        """Dora packed ``rgb`` (isolation / bag) → unpack → encode."""
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
            s0 = int(meta.get("stamp_sec", 0) or 0)
            ns0 = int(meta.get("stamp_nanosec", 0) or 0)
            stamp_sec = [s0] * n
            stamp_nanosec = [ns0] * n
        self._encode_and_send(
            rgbs,
            frame_ids[:n],
            stamp_sec[:n],
            stamp_nanosec[:n],
            src_meta=meta,
        )

    def run(self) -> None:
        try:
            if self._ros is not None:
                self._spin_running = True
                self._spin_thread = threading.Thread(
                    target=self._ros_spin_loop,
                    name="clipseg_ros_spin",
                    daemon=True,
                )
                self._spin_thread.start()
                self._log("ROS spin thread started")

            for event in self.dora:
                if event is None or event.get("type") == "STOP":
                    break
                if event.get("type") != "INPUT":
                    continue
                eid = event.get("id")

                if self.ros_io and eid == "tick":
                    self._drain_pending_encode()
                elif eid == "rgb" and not self.ros_io:
                    self._on_rgb(event)
        finally:
            self._spin_running = False
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=1.0)
                self._spin_thread = None
            if self._ros is not None:
                import rclpy

                self._ros.destroy_node()
                self._ros = None
                if rclpy.ok():
                    rclpy.shutdown()


def main():
    ClipSegEncoderNode(load_seg_config()).run()


if __name__ == "__main__":
    main()
