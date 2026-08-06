#!/usr/bin/env python3
"""Dora node: CLIPSeg vision encoder (single-cam or multi-cam batch).

Single-cam: ROS ``color_topic`` → features [L,S,D] + rgb.
Multi-cam: ROS ``color_topics`` list → pack N RGB, encode [N,L,S,D], Dora meta
includes ``n_cams`` / ``frame_ids`` / per-cam stamps.
"""

from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node as RosNode
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import torch

from dora import Node

from perception_models.clipseg.shared import (
    ClipSegParts,
    load_seg_config,
    prompt_layout_from_config,
)
from perception_models.cuda_ipc import cuda_ipc_supported, pack_arrow_f16, pack_cuda_tensor
from perception_models.multicam_rgb_batch import pack_rgb_batch


def main():
    cfg = load_seg_config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; clipseg encoder needs a GPU")
    torch.set_float32_matmul_precision("high")

    dora = Node()

    rclpy.init()
    ros = RosNode("dora_clipseg_encoder")
    bridge = CvBridge()

    color_topics = [str(t) for t in (cfg.get("color_topics") or []) if str(t)]
    if not color_topics:
        color_topics = [str(cfg.get("color_topic", "/camera/color/image_raw"))]
    multicam = len(color_topics) > 1
    latest = {t: None for t in color_topics}

    def _make_cb(topic: str):
        def _on_image(msg: Image):
            latest[topic] = msg

        return _on_image

    for t in color_topics:
        ros.create_subscription(Image, t, _make_cb(t), qos_profile_sensor_data)

    def log(msg: str):
        ros.get_logger().info(msg)

    layout = prompt_layout_from_config(cfg)
    trt_engine_path = str(cfg.get("clipseg_vision_engine", "") or "").strip()
    use_trt = bool(trt_engine_path)
    batch_n = int(cfg.get("clipseg_batch_n", len(color_topics)))

    parts = ClipSegParts(
        str(cfg.get("model_name", "CIDAS/clipseg-rd16")),
        layout,
        input_res=int(cfg.get("input_res", 224)),
        compile_mode=("none" if use_trt else str(cfg.get("compile_mode", "reduce-overhead"))),
        device="cuda",
        logger=log,
    )

    enc = None
    trt_enc = None
    if use_trt:
        from perception_models.clipseg.trt import load_clipseg_vision_engine, trt_encode_hs

        log(f"Loading CLIPSeg vision TensorRT engine: {trt_engine_path}")
        trt_enc = load_clipseg_vision_engine(trt_engine_path)
        dummy = torch.zeros(1, 3, parts.res, parts.res, device="cuda")
        for _ in range(3):
            _ = trt_encode_hs(trt_enc, dummy)
        torch.cuda.synchronize()
        log("CLIPSeg TRT vision warmup complete.")
    else:
        enc = parts.build_encoder()
        enc = parts.warmup_encoder(enc)

    profile = bool(cfg.get("profile", True))
    profile_every = max(1, int(cfg.get("profile_every", 30)))
    prof = {}
    prof_n = 0

    use_cuda_ipc = cuda_ipc_supported("cuda") and not multicam
    feature_mode = "torch_cuda_ipc" if use_cuda_ipc else "arrow_f16"
    if use_cuda_ipc:
        log("feature handoff: PyTorch CUDA IPC (GPU->GPU)")
    else:
        log(
            "feature handoff: Arrow float16 "
            + ("(multicam batch)" if multicam else "(CUDA IPC unsupported / disabled)")
        )

    enc_tag = "trt" if use_trt else "torch"
    log(
        f"dora clipseg_encoder up: topics={color_topics} multicam={multicam} "
        f"batch_n={batch_n} encode={enc_tag}"
    )

    for event in dora:
        if event is None:
            break
        if event.get("type") == "STOP":
            break
        if event.get("type") != "INPUT" or event.get("id") != "tick":
            continue

        for _ in range(8):
            rclpy.spin_once(ros, timeout_sec=0.0)

        msgs = [latest[t] for t in color_topics]
        if any(m is None for m in msgs):
            continue
        for t in color_topics:
            latest[t] = None

        try:
            rgbs = [
                np.ascontiguousarray(bridge.imgmsg_to_cv2(m, desired_encoding="rgb8"))
                for m in msgs
            ]
        except Exception as e:  # noqa: BLE001
            ros.get_logger().warn(f"cv_bridge conversion failed: {e}")
            continue

        h, w = rgbs[0].shape[:2]
        if profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if use_trt:
            from perception_models.clipseg.trt import trt_encode_hs

            pix = parts.preprocess_rgb_batch(rgbs)
            hs = trt_encode_hs(trt_enc, pix)
            if hs.dim() == 3:
                hs = hs.unsqueeze(0)
        else:
            hs = parts.encode_batch(rgbs, enc)

        if profile:
            torch.cuda.synchronize()
            prof["clipseg_encode"] = (
                prof.get("clipseg_encode", 0.0) + (time.perf_counter() - t0) * 1000.0
            )

        n_cams = int(hs.shape[0])
        n_layers, seq_len, hidden = (
            int(hs.shape[1]),
            int(hs.shape[2]),
            int(hs.shape[3]),
        )

        frame_ids = [str(m.header.frame_id or "") for m in msgs]
        stamp_sec = [int(m.header.stamp.sec) for m in msgs]
        stamp_nanosec = [int(m.header.stamp.nanosec) for m in msgs]

        meta = {
            "n_cams": n_cams,
            "stamp_sec_list": stamp_sec,
            "stamp_nanosec_list": stamp_nanosec,
            "frame_ids": frame_ids,
            "stamp_sec": stamp_sec[0],
            "stamp_nanosec": stamp_nanosec[0],
            "frame_id": frame_ids[0],
            "height": int(h),
            "width": int(w),
            "n_layers": n_layers,
            "seq_len": seq_len,
            "hidden": hidden,
            "encode_backend": enc_tag,
            "feature_batch": n_cams,
        }

        if profile:
            t_ipc = time.perf_counter()

        rgb_arr, rgb_meta = pack_rgb_batch(
            rgbs, frame_ids=frame_ids, stamp_sec=stamp_sec, stamp_nanosec=stamp_nanosec
        )
        dora.send_output("rgb", rgb_arr, {**meta, **rgb_meta})

        if use_cuda_ipc and n_cams == 1:
            try:
                buf, ipc_meta, mode = pack_cuda_tensor(hs[0])
                feature_mode = mode
                out_meta = {**meta, **ipc_meta, "feature_mode": mode}
                dora.send_output("features", buf, out_meta)
            except Exception as e:  # noqa: BLE001
                ros.get_logger().warn(
                    f"CUDA IPC failed at runtime ({e}); switching to Arrow float16"
                )
                use_cuda_ipc = False
                feature_mode = "arrow_f16"

        if not use_cuda_ipc or n_cams > 1:
            feature_mode = "arrow_f16"
            dora.send_output(
                "features",
                pack_arrow_f16(hs),
                {**meta, "feature_mode": "arrow_f16", "feature_batch": n_cams},
            )

        if profile:
            prof["feature_xfer"] = (
                prof.get("feature_xfer", 0.0) + (time.perf_counter() - t_ipc) * 1000.0
            )
            prof_n += 1
            if prof_n >= profile_every:
                parts_s = " ".join(
                    f"{k}={v / prof_n:5.1f}" for k, v in sorted(prof.items())
                )
                log(
                    f"[timing ms/frame n={prof_n} mode={feature_mode} cams={n_cams}] {parts_s}"
                )
                prof, prof_n = {}, 0

    ros.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
