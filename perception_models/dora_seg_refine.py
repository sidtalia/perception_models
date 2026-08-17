#!/usr/bin/env python3
"""Dora-only: ClipSeg FiLM ∥ NanoSAM encode (CUDA streams), then SAM decode.

No ROS. Optional Dora viz (``dora_viz``). E2E = done − ``t_src_perf``.

For N cams: enqueue ``decode_probs_batch`` on stream_film and
``encode_tensor`` on stream_sam so they overlap; then ``decode_multi``.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F

from dora import Node

from perception_models.clipseg.shared import (
    ClipSegParts,
    load_seg_config,
    prompt_layout_from_config,
)
from perception_models.clipseg.trt import TrtFilmDecoder, load_clipseg_film_engine
from perception_models.cuda_ipc import unpack_arrow_f16, unpack_cuda_tensor
from perception_models.multicam_rgb_batch import (
    camera_ns_from_color_topic,
    refined_people_mask_topic,
    refined_trav_topic,
    unpack_rgb_batch,
)
from perception_models.nanosam_trt import NanoSamDecoder, NanoSamEncoder
from perception_models.sam_box_merge import clipseg_nav_boxes
from perception_models.seg_ops import (
    blend_overlay,
    labels_to_boxes_batch,
    labels_to_color,
    pack_traversability_rgb,
    traversability_colors_from_config,
)


def _feature_shape(meta: dict) -> tuple[int, ...]:
    n_batch = int(meta.get("feature_batch", meta.get("n_cams", 1)))
    base = (
        int(meta["n_layers"]),
        int(meta["seq_len"]),
        int(meta["hidden"]),
    )
    if n_batch > 1:
        return (n_batch, *base)
    return base


def _open_features(event, *, non_blocking: bool = False) -> torch.Tensor:
    meta = event["metadata"]
    mode = str(meta.get("feature_mode", "arrow_f16"))
    shape = _feature_shape(meta)

    if mode == "torch_cuda_ipc":
        return unpack_cuda_tensor(event["value"], meta).reshape(*shape).contiguous()
    if mode == "cuda_ipc":
        from dora.cuda import ipc_buffer_to_ipc_handle, open_ipc_handle

        handle = ipc_buffer_to_ipc_handle(event["value"], meta)
        with open_ipc_handle(handle, meta) as tensor:
            return tensor.reshape(*shape).contiguous().clone()
    return unpack_arrow_f16(event["value"], shape, non_blocking=non_blocking)


def _cam_short_name(frame_id: str, index: int) -> str:
    fid = str(frame_id or "")
    for token in ("front", "left", "right"):
        if token in fid:
            return token
    return f"cam{index}"


def _union(refined: torch.Tensor, idxs) -> torch.Tensor:
    sel = [i for i in idxs if 0 <= i < refined.shape[0]]
    if not sel:
        return refined.new_zeros(refined.shape[1:])
    return refined[sel].amax(dim=0)


def _t_src_perf(meta: dict) -> float | None:
    v = meta.get("t_src_perf")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _e2e_stats(samples: list[float]) -> dict:
    if not samples:
        return {
            "n": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "samples_ms": [],
        }
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "samples_ms": [float(x) for x in samples],
    }


def _stamp_scalar(v, default: int = 0) -> int:
    if isinstance(v, (list, tuple)):
        v = v[0] if v else default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _scale_boxes_xyxy(boxes, sx: float, sy: float):
    out = []
    for box in boxes:
        x0, y0, x1, y1 = box
        out.append((x0 * sx, y0 * sy, x1 * sx, y1 * sy))
    return out


class _PrintLogger:
    _prefix = "dora_seg_refine"

    def info(self, msg: str) -> None:
        print(f"[{self._prefix}] {msg}", flush=True)

    def warning(self, msg: str) -> None:
        print(f"[{self._prefix}] WARNING: {msg}", flush=True)

    def warn(self, msg: str) -> None:
        self.warning(msg)

    def get_logger(self) -> _PrintLogger:
        return self


class SegRefineNode:
    """FiLM TRT ∥ NanoSAM encode → decode_multi; optional async Dora viz."""

    def __init__(self, cfg: dict):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; seg_refine needs a GPU")
        torch.set_float32_matmul_precision("high")

        self.cfg = cfg
        self.dora = Node()
        self.logger = _PrintLogger()
        self.log = self.logger.info

        self.dora_viz = bool(cfg.get("dora_viz", False))
        self.e2e_samples: list[float] = []
        self.e2e_log = bool(cfg.get("e2e_log", True))
        self.e2e_log_every_s = max(1.0, float(cfg.get("e2e_log_every_s", 60.0)))
        self.e2e_log_t0 = 0.0

        self.layout = prompt_layout_from_config(cfg)
        batch_n = max(1, int(cfg.get("clipseg_batch_n", 1)))
        if bool(cfg.get("fanout_identical", False)):
            batch_n = max(
                batch_n, len([n for n in (cfg.get("camera_names") or []) if n])
            )
        self.batch_n = batch_n

        self.parts = ClipSegParts(
            str(cfg.get("model_name", "CIDAS/clipseg-rd16")),
            self.layout,
            input_res=int(cfg.get("input_res", 224)),
            compile_mode="none",
            device="cuda",
            logger=self.log,
        )
        self.dec = self._load_film_decoder()

        self.aggregate = str(cfg.get("aggregate", "max")).lower()
        self.threshold = float(cfg.get("threshold", 0.3))
        self.trav_pack_threshold = float(
            cfg.get("traversability_pack_threshold", self.threshold)
        )
        self.trav_colors = traversability_colors_from_config(cfg)
        self.group_has_prompts = [len(idx) > 0 for idx in self.layout.group_index]
        self.viz_overlay = bool(cfg.get("viz_overlay", False))
        self.overlay_alpha = float(cfg.get("overlay_alpha", 0.5))
        self.sam_overlay = bool(cfg.get("sam_overlay", False))
        self.viz_send_trav = bool(cfg.get("viz_send_trav", not self.sam_overlay))
        self.viz_min_interval_ms = float(cfg.get("viz_min_interval_ms", 0) or 0)
        self.last_viz_t = 0.0
        self.viz_pool = (
            ThreadPoolExecutor(max_workers=1) if self.dora_viz else None
        )
        self.viz_fut = None

        self.positive_groups = [int(x) for x in cfg.get("positive_groups", [0])]
        self.negative_groups = [int(x) for x in cfg.get("negative_groups", [1])]
        _pg = cfg.get("people_groups", None)
        if _pg is None:
            _pg = [] if cfg.get("yolo_owns_people") else [2]
        self.people_groups = [int(x) for x in _pg]
        # SAM/CLIPSeg channel count follows prompt layout only (YOLO people are separate).
        self.n_groups = int(self.layout.n_groups)
        self.people_group_idx = (
            int(self.people_groups[0])
            if self.people_groups
            else max(0, self.n_groups - 1)
        )

        enc_path = str(cfg.get("sam_image_encoder", "") or "").strip()
        dec_path = str(cfg.get("sam_mask_decoder", "") or "").strip()
        if not enc_path or not dec_path:
            raise RuntimeError(
                "sam_image_encoder and sam_mask_decoder required (batched TRT)"
            )

        self.sam_min_area = float(cfg.get("sam_min_area", 0.002))
        self.sam_max_boxes_nav = int(cfg.get("sam_max_boxes", 6))
        sam_dec_max_k = max(
            self.sam_max_boxes_nav,
            self.sam_max_boxes_nav * max(1, batch_n),
            int(cfg.get("sam_decoder_max_batch", 16)),
        )
        self.profile = (
            os.environ.get("PROFILE", "0") == "1" or bool(cfg.get("profile", False))
        )
        if self.profile:
            os.environ["PROFILE"] = "1"

        self.log("Loading NanoSAM (TensorRT) ...")
        self.sam_encoder = NanoSamEncoder(
            enc_path,
            image_encoder_size=int(cfg.get("sam_image_size", 512)),
            device="cuda",
            logger=self.logger,
        )
        self.sam_decoder = NanoSamDecoder(
            dec_path,
            orig_image_encoder_size=int(cfg.get("sam_image_size", 512)),
            device="cuda",
            logger=self.logger,
            max_batch_k=sam_dec_max_k,
        )
        self.stream_film = torch.cuda.Stream()
        self.stream_sam = torch.cuda.Stream()

        # Dora edges are queue_size=1; hold at most one unmatched rgb / features.
        self.pending_rgb: tuple | None = None
        self.pending_feat: tuple | None = None

        # ROS egress: per-cam publishers keyed by camera namespace (from color_topics meta).
        self.ros_io = bool(cfg.get("ros_io", False)) or bool(
            cfg.get("use_ros_edge", False)
        )
        self._ros = None
        self._bridge = None
        self._trav_pubs: dict[str, object] = {}
        self._people_pubs: dict[str, object] = {}
        if self.ros_io:
            self._setup_ros_publishers()

        camera_names = [str(n) for n in (cfg.get("camera_names") or []) if str(n)]
        self.log(
            "dora seg_refine up FiLM∥SAM-encode → decode_multi"
            + f" sam_boxes={self.sam_max_boxes_nav}"
            + (f" cams={camera_names}" if camera_names else "")
            + f" dora_viz={'on' if self.dora_viz else 'off'}"
            + f" ros_io={'on' if self.ros_io else 'off'}"
        )

    def _load_film_decoder(self) -> TrtFilmDecoder:
        path = str(self.cfg.get("clipseg_film_engine", "") or "").strip()
        assert path, (
            "clipseg_film_engine required (TRT-only FiLM); build with "
            "scripts/build_clipseg_film_engine.sh"
        )
        assert os.path.isfile(path), f"clipseg_film_engine not found: {path}"
        self.log(f"Loading CLIPSeg FiLM TRT: {path}")
        film_eng = load_clipseg_film_engine(path)
        dec = TrtFilmDecoder(film_eng)
        with torch.inference_mode():
            try:
                hs_shape = [
                    int(d) for d in film_eng.engine.get_tensor_shape("stacked_hs")
                ]
            except Exception:  # noqa: BLE001
                hs_shape = [
                    self.batch_n,
                    len(self.parts.extract),
                    (self.parts.res // 16) ** 2 + 1,
                    768,
                ]
            if hs_shape[0] <= 0:
                hs_shape[0] = self.batch_n
            dummy_hs = torch.zeros(*hs_shape, device="cuda", dtype=torch.float32)
            _ = dec(dummy_hs, self.parts.cond)
            del dummy_hs, _
        torch.cuda.synchronize()
        self.log("CLIPSeg FiLM TRT warmup complete.")
        return dec

    def _setup_ros_publishers(self) -> None:
        """Lazy pubs created on first frame from ``color_topics`` meta."""
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node as RosNode

        if not rclpy.ok():
            rclpy.init(args=None)
        self._ros = RosNode("dora_seg_refine")
        self._bridge = CvBridge()
        self.log("ROS egress ready (per-cam pubs from color_topics meta)")

    def _trav_pub_for(self, color_topic: str):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        ns = camera_ns_from_color_topic(color_topic)
        topic = refined_trav_topic(ns)
        pub = self._trav_pubs.get(topic)
        if pub is None and self._ros is not None:
            pub = self._ros.create_publisher(Image, topic, qos_profile_sensor_data)
            self._trav_pubs[topic] = pub
            self.log(f"ROS trav pub: {topic}")
        return pub

    def _people_pub_for(self, color_topic: str):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        ns = camera_ns_from_color_topic(color_topic)
        topic = refined_people_mask_topic(ns)
        pub = self._people_pubs.get(topic)
        if pub is None and self._ros is not None:
            pub = self._ros.create_publisher(Image, topic, qos_profile_sensor_data)
            self._people_pubs[topic] = pub
            self.log(f"ROS people_mask pub: {topic}")
        return pub

    @staticmethod
    def _header_from_meta(meta: dict, index: int):
        from builtin_interfaces.msg import Time
        from std_msgs.msg import Header

        secs = meta.get("stamp_sec_list") or [meta.get("stamp_sec", 0)]
        nsecs = meta.get("stamp_nanosec_list") or [meta.get("stamp_nanosec", 0)]
        frames = meta.get("frame_ids") or [meta.get("frame_id", "")]
        i = min(index, max(0, len(secs) - 1))
        j = min(index, max(0, len(nsecs) - 1))
        k = min(index, max(0, len(frames) - 1))
        hdr = Header()
        hdr.stamp = Time(sec=int(secs[i] or 0), nanosec=int(nsecs[j] or 0))
        hdr.frame_id = str(frames[k] or "")
        return hdr

    def _publish_ros_trav(
        self,
        meta_rgb: dict,
        sam_nav_all: torch.Tensor,
        *,
        n_cams: int,
    ) -> None:
        """Publish refined trav (+ people) per cam using ingress topic/frame meta."""
        if not self.ros_io or self._ros is None or self._bridge is None:
            return
        topics = [str(t) for t in (meta_rgb.get("color_topics") or [])]
        if len(topics) < n_cams:
            self.log(
                "WARN: color_topics meta missing/short; "
                "set encoder ros_io color_topics for per-cam egress"
            )
            return

        for i in range(n_cams):
            color_topic = topics[i]
            pub = self._trav_pub_for(color_topic)
            if pub is None:
                continue
            hdr = self._header_from_meta(meta_rgb, i)
            rgb_np, _ = self._pack_trav_rgb(sam_nav_all[i])
            trav_msg = self._bridge.cv2_to_imgmsg(rgb_np, encoding="rgb8")
            trav_msg.header = hdr
            pub.publish(trav_msg)

            people = _union(sam_nav_all[i], self.people_groups)
            people_np = (people.detach().float().cpu().numpy() >= 0.5).astype(
                np.uint8
            ) * 255
            ppub = self._people_pub_for(color_topic)
            if ppub is not None:
                pmsg = self._bridge.cv2_to_imgmsg(people_np, encoding="mono8")
                pmsg.header = hdr
                ppub.publish(pmsg)

        import rclpy

        rclpy.spin_once(self._ros, timeout_sec=0.0)

    def _record_e2e(self, meta_rgb: dict) -> float | None:
        t0 = _t_src_perf(meta_rgb)
        if t0 is None:
            return None
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        self.e2e_samples.append(e2e_ms)
        now = time.perf_counter()
        if self.e2e_log and now - self.e2e_log_t0 >= self.e2e_log_every_s:
            st = _e2e_stats(self.e2e_samples)
            self.log(
                f"[e2e ms] n={st['n']} median={st['median_ms']:.1f} "
                f"mean={st['mean_ms']:.1f} p90={st['p90_ms']:.1f} last={e2e_ms:.1f}"
            )
            self.e2e_log_t0 = now
            e2e_path = str(self.cfg.get("e2e_latency_json", "") or "").strip()
            if e2e_path:
                with open(e2e_path, "w", encoding="utf-8") as f:
                    json.dump(st, f, indent=2)
        return e2e_ms

    def _dora_image_meta(
        self, meta_rgb: dict, *, e2e_ms: float | None = None
    ) -> dict:
        out = {
            "width": int(meta_rgb["width"]),
            "height": int(meta_rgb["height"]),
            "encoding": "rgb8",
            "n_cams": int(meta_rgb.get("n_cams", 1)),
            "frame_id": str(meta_rgb.get("frame_id", "")),
        }
        t0 = _t_src_perf(meta_rgb)
        if t0 is not None:
            out["t_src_perf"] = t0
        if e2e_ms is not None:
            out["e2e_ms"] = float(e2e_ms)
        return out

    def _send_dora_viz(
        self,
        pending: dict[str, np.ndarray] | None,
        meta_rgb: dict,
        *,
        e2e_ms: float | None,
    ) -> None:
        if not self.dora_viz or not pending:
            return
        meta_base = self._dora_image_meta(meta_rgb, e2e_ms=e2e_ms)
        if "trav" in pending:
            trav = np.ascontiguousarray(pending["trav"], dtype=np.uint8)
            self.dora.send_output(
                "trav",
                pa.array(trav.reshape(-1), type=pa.uint8()),
                meta_base,
            )
        if "overlay" in pending:
            ov = np.ascontiguousarray(pending["overlay"], dtype=np.uint8)
            self.dora.send_output(
                "overlay",
                pa.array(ov.reshape(-1), type=pa.uint8()),
                meta_base,
            )

    def _pack_trav_rgb(self, group_maps):
        return pack_traversability_rgb(
            group_maps,
            positive_groups=self.positive_groups,
            negative_groups=self.negative_groups,
            people_groups=self.people_groups,
            group_has_prompts=self.group_has_prompts,
            colors=self.trav_colors,
            threshold=self.trav_pack_threshold,
        )

    def _nav_reqs_from_labels_batch(self, labels_np_all, h_full, w_full):
        hl = int(labels_np_all.shape[1])
        wl = int(labels_np_all.shape[2])
        sx = float(w_full) / float(wl)
        sy = float(h_full) / float(hl)
        cand_lists = labels_to_boxes_batch(
            labels_np_all,
            self.n_groups,
            self.sam_min_area,
            max(self.sam_max_boxes_nav * 2, self.sam_max_boxes_nav),
        )
        out = []
        for cands in cand_lists:
            nav_classes, nav_boxes = clipseg_nav_boxes(
                cands,
                people_group_idx=self.people_group_idx,
                drop_people=False,
                max_boxes=self.sam_max_boxes_nav,
            )
            if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
                nav_boxes = _scale_boxes_xyxy(nav_boxes, sx, sy)
            n_nav = max(
                [
                    *nav_classes,
                    *self.positive_groups,
                    *self.negative_groups,
                    *self.people_groups,
                ],
                default=0,
            ) + 1
            out.append((nav_classes, nav_boxes, n_nav, (h_full, w_full)))
        return out

    @staticmethod
    def _upsample_groups(groups_1ghw: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = groups_1ghw.unsqueeze(0)
        if x.dtype != torch.float16:
            x = x.to(dtype=torch.float16)
        return F.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=False
        ).clamp_(0.0, 1.0)[0]

    def _front_viz(
        self,
        rgb,
        probs_lr,
        labels_np,
        conf_np,
        refined_nav,
        people_mask,
        *,
        h: int,
        w: int,
    ) -> dict[str, np.ndarray] | None:
        if not self.dora_viz:
            return None
        pending: dict[str, np.ndarray] = {}
        need_film_full = refined_nav is None or self.viz_overlay
        probs_full = None
        if need_film_full and probs_lr is not None:
            probs_full = self._upsample_groups(probs_lr, h, w).float()
        if self.viz_overlay:
            if labels_np.shape[0] != h or labels_np.shape[1] != w:
                labels_np = cv2.resize(
                    labels_np, (w, h), interpolation=cv2.INTER_NEAREST
                )
                conf_np = cv2.resize(
                    conf_np.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
                )
            color = labels_to_color(labels_np, self.layout.n_groups)
            pending["overlay"] = blend_overlay(
                rgb, color, conf_np, self.overlay_alpha
            )
        pack_src = refined_nav if refined_nav is not None else probs_full
        if pack_src is None:
            return pending or None
        rgb_np, nontrav = self._pack_trav_rgb(pack_src)
        if self.viz_send_trav:
            pending["trav"] = rgb_np
        if self.sam_overlay and refined_nav is not None:
            conf = nontrav.detach().float().cpu().numpy()
            ov = blend_overlay(rgb, rgb_np, conf, self.overlay_alpha)
            if people_mask is not None:
                pm = people_mask.detach().float().cpu().numpy() >= 0.5
                ov = ov.copy()
                ov[pm, 2] = np.maximum(ov[pm, 2], 200)
            pending["overlay"] = ov
        elif not pending and not self.viz_send_trav:
            pending["trav"] = rgb_np
        return pending or None

    def _run_matched(self, meta_rgb: dict, rgbs: list, feat_event) -> None:
        n_cams = len(rgbs)
        h = int(meta_rgb["height"])
        w = int(meta_rgb["width"])

        hs = _open_features(feat_event, non_blocking=True)
        if hs.dim() == 3:
            hs = hs.unsqueeze(0)

        t0 = time.perf_counter()
        with torch.cuda.stream(self.stream_film):
            probs_all = self.parts.decode_probs_batch(
                hs, self.dec, aggregate=self.aggregate
            )
        sam_input = self.sam_encoder.preprocess_batch(rgbs)
        with torch.cuda.stream(self.stream_sam):
            sam_feats_all = self.sam_encoder.encode_tensor(
                sam_input, synchronize=False
            )
        with torch.cuda.stream(self.stream_film):
            maxv, arg = probs_all.max(dim=1)
            labels_u8 = torch.where(
                maxv >= self.threshold,
                (arg + 1).to(torch.uint8),
                torch.zeros_like(arg, dtype=torch.uint8),
            )
            labels_np_all = labels_u8.cpu().numpy()
            conf_np_all = maxv.float().cpu().numpy()
        nav_reqs = self._nav_reqs_from_labels_batch(labels_np_all, h, w)
        self.stream_sam.synchronize()
        t1 = time.perf_counter()
        if self.profile:
            print(f"FiLM||SAM encode time: {(t1 - t0) * 1000.0} ms", flush=True)

        t2 = time.perf_counter()
        sam_nav_all = self.sam_decoder.decode_multi(sam_feats_all, nav_reqs)
        t3 = time.perf_counter()
        if self.profile:
            tim = self.sam_decoder.last_timing
            print(
                f"SAM decode time: {(t3 - t2) * 1000.0} ms "
                f"(trt={tim.get('trt_ms', 0):.1f} upscale={tim.get('upscale_ms', 0):.1f} "
                f"boxes={tim.get('n_boxes', 0)} upscales={tim.get('n_upscales', 0)})",
                flush=True,
            )

        e2e_ms = self._record_e2e(meta_rgb)

        # ROS product egress (mapper): all cams, frames/topics from encoder meta.
        self._publish_ros_trav(meta_rgb, sam_nav_all, n_cams=n_cams)

        if not self.dora_viz or self.viz_pool is None:
            return

        now = time.perf_counter()
        if (
            self.viz_min_interval_ms > 0
            and (now - self.last_viz_t) * 1000.0 < self.viz_min_interval_ms
        ):
            return
        if self.viz_fut is not None and not self.viz_fut.done():
            return

        frame_ids = list(meta_rgb.get("frame_ids") or [])
        front_i = 0
        for i, fid in enumerate(frame_ids):
            if _cam_short_name(str(fid), i) == "front":
                front_i = i
                break
        refined_front = sam_nav_all[front_i]
        people_front = _union(refined_front, self.people_groups)
        labels_np = np.ascontiguousarray(labels_np_all[front_i])
        conf_np = np.ascontiguousarray(conf_np_all[front_i])
        rgb_front = np.ascontiguousarray(rgbs[front_i])
        maps_cpu = refined_front.detach().to(dtype=torch.float32, device="cpu")
        people_cpu = people_front.detach().to(dtype=torch.float32, device="cpu")
        probs_lr_cpu = (
            probs_all[front_i].detach().to(dtype=torch.float16, device="cpu")
            if self.viz_overlay
            else None
        )
        cam_meta = dict(meta_rgb)
        cam_meta["n_cams"] = n_cams
        if frame_ids:
            cam_meta["frame_id"] = str(frame_ids[front_i])

        def _viz_job():
            pending = self._front_viz(
                rgb_front,
                probs_lr_cpu,
                labels_np,
                conf_np,
                maps_cpu,
                people_cpu,
                h=h,
                w=w,
            )
            self._send_dora_viz(pending, cam_meta, e2e_ms=e2e_ms)

        self.viz_fut = self.viz_pool.submit(_viz_job)
        self.last_viz_t = now
        if self.profile:
            print("viz staged async (drop-if-busy)", flush=True)

    @staticmethod
    def _drop_feat(slot: tuple | None) -> None:
        if slot is None:
            return
        try:
            _ = _open_features(slot[1])
            del _
        except Exception:  # noqa: BLE001
            pass

    def _on_event(self, event) -> None:
        eid = event["id"]
        meta = event["metadata"]
        stamp_key = (
            _stamp_scalar(meta.get("stamp_sec", meta.get("stamp_sec0", 0))),
            _stamp_scalar(meta.get("stamp_nanosec", meta.get("stamp_nanosec0", 0))),
        )

        if eid == "rgb":
            rgbs = unpack_rgb_batch(event["value"], meta)
            self.pending_rgb = (stamp_key, meta, rgbs)
        elif eid == "features":
            self._drop_feat(self.pending_feat)
            self.pending_feat = (stamp_key, event)
        else:
            return

        if (
            self.pending_rgb is not None
            and self.pending_feat is not None
            and self.pending_rgb[0] == self.pending_feat[0]
        ):
            _, meta_rgb, rgbs = self.pending_rgb
            _, feat_event = self.pending_feat
            self.pending_rgb = None
            self.pending_feat = None
            self._run_matched(meta_rgb, rgbs, feat_event)

    def _shutdown(self) -> None:
        if self.viz_pool is not None:
            if self.viz_fut is not None:
                try:
                    self.viz_fut.result(timeout=2.0)
                except Exception:  # noqa: BLE001
                    pass
            self.viz_pool.shutdown(wait=False)

        if self._ros is not None:
            import rclpy

            self._ros.destroy_node()
            self._ros = None
            if rclpy.ok():
                rclpy.shutdown()

        e2e_path = str(self.cfg.get("e2e_latency_json", "") or "").strip()
        if self.e2e_samples:
            st = _e2e_stats(self.e2e_samples)
            self.log(
                f"[e2e ms exit] n={st['n']} median={st['median_ms']:.1f} "
                f"mean={st['mean_ms']:.1f} p90={st['p90_ms']:.1f} "
                f"min={st['min_ms']:.1f} max={st['max_ms']:.1f}"
            )
            if e2e_path:
                with open(e2e_path, "w", encoding="utf-8") as f:
                    json.dump(st, f, indent=2)
                self.log(f"e2e latency JSON -> {e2e_path}")

    def run(self) -> None:
        try:
            for event in self.dora:
                if event is None or event.get("type") == "STOP":
                    break
                if event.get("type") != "INPUT":
                    continue
                self._on_event(event)
        finally:
            self._shutdown()


def main():
    SegRefineNode(load_seg_config()).run()


if __name__ == "__main__":
    main()
