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

from dora import Node

from perception_models.clipseg.shared import (
    ClipSegParts,
    load_seg_config,
    prompt_layout_from_config,
)
from perception_models.clipseg.trt import TrtFilmDecoder, load_clipseg_film_engine
from perception_models.cuda_ipc import unpack_arrow_f16, unpack_cuda_tensor
from perception_models.multicam_rgb_batch import unpack_rgb_batch
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


def main():
    cfg = load_seg_config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; seg_refine needs a GPU")
    torch.set_float32_matmul_precision("high")

    dora = Node()
    logger = _PrintLogger()
    log = logger.info

    dora_viz = bool(cfg.get("dora_viz", False))
    e2e_samples: list[float] = []
    e2e_log_n = 0
    e2e_log_every = max(1, int(cfg.get("e2e_log_every", 10)))

    def _record_e2e(meta_rgb: dict) -> float | None:
        nonlocal e2e_log_n
        t0 = _t_src_perf(meta_rgb)
        if t0 is None:
            return None
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        e2e_samples.append(e2e_ms)
        e2e_log_n += 1
        if e2e_log_n >= e2e_log_every:
            st = _e2e_stats(e2e_samples)
            log(
                f"[e2e ms] n={st['n']} median={st['median_ms']:.1f} "
                f"mean={st['mean_ms']:.1f} p90={st['p90_ms']:.1f} last={e2e_ms:.1f}"
            )
            e2e_log_n = 0
            e2e_path = str(cfg.get("e2e_latency_json", "") or "").strip()
            if e2e_path:
                with open(e2e_path, "w", encoding="utf-8") as f:
                    json.dump(st, f, indent=2)
        return e2e_ms

    def _dora_image_meta(meta_rgb: dict, *, e2e_ms: float | None = None) -> dict:
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
        pending: dict[str, np.ndarray] | None,
        meta_rgb: dict,
        *,
        e2e_ms: float | None,
    ) -> None:
        if not dora_viz or not pending:
            return
        meta_base = _dora_image_meta(meta_rgb, e2e_ms=e2e_ms)
        if "trav" in pending:
            trav = np.ascontiguousarray(pending["trav"], dtype=np.uint8)
            dora.send_output(
                "trav",
                pa.array(trav.reshape(-1), type=pa.uint8()),
                meta_base,
            )
        if "overlay" in pending:
            ov = np.ascontiguousarray(pending["overlay"], dtype=np.uint8)
            dora.send_output(
                "overlay",
                pa.array(ov.reshape(-1), type=pa.uint8()),
                meta_base,
            )

    layout = prompt_layout_from_config(cfg)
    batch_n = max(1, int(cfg.get("clipseg_batch_n", 1)))
    if bool(cfg.get("fanout_identical", False)):
        batch_n = max(batch_n, len([n for n in (cfg.get("camera_names") or []) if n]))
    parts = ClipSegParts(
        str(cfg.get("model_name", "CIDAS/clipseg-rd16")),
        layout,
        input_res=int(cfg.get("input_res", 224)),
        compile_mode="none",
        device="cuda",
        logger=log,
    )
    film_engine_path = str(cfg.get("clipseg_film_engine", "") or "").strip()
    assert film_engine_path, (
        "clipseg_film_engine required (TRT-only FiLM); build with "
        "scripts/build_clipseg_film_engine.sh"
    )
    assert os.path.isfile(film_engine_path), (
        f"clipseg_film_engine not found: {film_engine_path}"
    )
    log(f"Loading CLIPSeg FiLM TRT: {film_engine_path}")
    film_eng = load_clipseg_film_engine(film_engine_path)
    dec = TrtFilmDecoder(film_eng)
    # Touch once so binding / context alloc is off the hot path.
    with torch.inference_mode():
        try:
            hs_shape = [int(d) for d in film_eng.engine.get_tensor_shape("stacked_hs")]
        except Exception:  # noqa: BLE001
            hs_shape = [
                batch_n,
                len(parts.extract),
                (parts.res // 16) ** 2 + 1,
                768,
            ]
        if hs_shape[0] <= 0:
            hs_shape[0] = batch_n
        dummy_hs = torch.zeros(*hs_shape, device="cuda", dtype=torch.float32)
        _ = dec(dummy_hs, parts.cond)
        del dummy_hs, _
    torch.cuda.synchronize()
    log("CLIPSeg FiLM TRT warmup complete.")

    aggregate = str(cfg.get("aggregate", "max")).lower()
    threshold = float(cfg.get("threshold", 0.3))
    trav_pack_threshold = float(cfg.get("traversability_pack_threshold", threshold))
    trav_colors = traversability_colors_from_config(cfg)
    group_has_prompts = [len(idx) > 0 for idx in layout.group_index]
    viz_overlay = bool(cfg.get("viz_overlay", False))
    overlay_alpha = float(cfg.get("overlay_alpha", 0.5))
    sam_overlay = bool(cfg.get("sam_overlay", False))
    # When sam_overlay paints trav into overlay, skip a second full-frame Dora hop
    # unless explicitly requested (halves Rerun / Arrow bandwidth).
    viz_send_trav = bool(cfg.get("viz_send_trav", not sam_overlay))
    viz_min_interval_ms = float(cfg.get("viz_min_interval_ms", 0) or 0)
    last_viz_t = 0.0
    # One worker: pack stays on the GPU thread; Dora→Rerun send can lag without
    # blocking the next frame (drop if previous send still in flight).
    viz_pool = ThreadPoolExecutor(max_workers=1) if dora_viz else None
    viz_fut = None

    positive_groups = [int(x) for x in cfg.get("positive_groups", [0])]
    negative_groups = [int(x) for x in cfg.get("negative_groups", [1])]
    people_groups = [int(x) for x in cfg.get("people_groups", [2])]
    n_groups = max(
        [*positive_groups, *negative_groups, *people_groups, layout.n_groups - 1],
        default=0,
    ) + 1
    people_group_idx = int(people_groups[0]) if people_groups else max(0, n_groups - 1)

    enc_path = str(cfg.get("sam_image_encoder", "") or "").strip()
    dec_path = str(cfg.get("sam_mask_decoder", "") or "").strip()
    if not enc_path or not dec_path:
        raise RuntimeError(
            "sam_image_encoder and sam_mask_decoder required (batched TRT)"
        )

    sam_min_area = float(cfg.get("sam_min_area", 0.002))
    sam_max_boxes_nav = int(cfg.get("sam_max_boxes", 6))
    # Cap decoder CUDA buffers (multicam × max boxes; TRT engines usually ≤16).
    sam_dec_max_k = max(
        sam_max_boxes_nav,
        sam_max_boxes_nav * max(1, batch_n),
        int(cfg.get("sam_decoder_max_batch", 16)),
    )
    profile = os.environ.get("PROFILE", "0") == "1" or bool(cfg.get("profile", False))
    if profile:
        os.environ["PROFILE"] = "1"
    log("Loading NanoSAM (TensorRT) ...")
    sam_encoder = NanoSamEncoder(
        enc_path,
        image_encoder_size=int(cfg.get("sam_image_size", 512)),
        device="cuda",
        logger=logger,
    )
    sam_decoder = NanoSamDecoder(
        dec_path,
        orig_image_encoder_size=int(cfg.get("sam_image_size", 512)),
        device="cuda",
        logger=logger,
        max_batch_k=sam_dec_max_k,
    )
    stream_film = torch.cuda.Stream()
    stream_sam = torch.cuda.Stream()

    # Dora edges are queue_size=1 (latest-only). We still hold at most one unmatched
    # rgb / features here because they arrive as separate INPUT events.
    pending_rgb: tuple | None = None  # (stamp_key, meta, rgbs)
    pending_feat: tuple | None = None  # (stamp_key, event)
    camera_names = [str(n) for n in (cfg.get("camera_names") or []) if str(n)]
    log(
        "dora seg_refine up (isolation) FiLM∥SAM-encode → decode_multi"
        + f" sam_boxes={sam_max_boxes_nav}"
        + (f" cams={camera_names}" if camera_names else "")
        + f" dora_viz={'on' if dora_viz else 'off'}"
    )

    def _pack_trav_rgb(group_maps):
        return pack_traversability_rgb(
            group_maps,
            positive_groups=positive_groups,
            negative_groups=negative_groups,
            people_groups=people_groups,
            group_has_prompts=group_has_prompts,
            colors=trav_colors,
            threshold=trav_pack_threshold,
        )

    def _scale_boxes_xyxy(boxes, sx: float, sy: float):
        out = []
        for box in boxes:
            x0, y0, x1, y1 = box
            out.append((x0 * sx, y0 * sy, x1 * sx, y1 * sy))
        return out

    def _nav_reqs_from_labels_batch(labels_np_all, h_full, w_full):
        # CC / boxes at FiLM map res; scale into full image for SAM.
        hl = int(labels_np_all.shape[1])
        wl = int(labels_np_all.shape[2])
        sx = float(w_full) / float(wl)
        sy = float(h_full) / float(hl)
        cand_lists = labels_to_boxes_batch(
            labels_np_all,
            n_groups,
            sam_min_area,
            max(sam_max_boxes_nav * 2, sam_max_boxes_nav),
        )
        out = []
        for cands in cand_lists:
            nav_classes, nav_boxes = clipseg_nav_boxes(
                cands,
                people_group_idx=people_group_idx,
                drop_people=False,
                max_boxes=sam_max_boxes_nav,
            )
            if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
                nav_boxes = _scale_boxes_xyxy(nav_boxes, sx, sy)
            n_nav = max(
                [*nav_classes, *positive_groups, *negative_groups, *people_groups],
                default=0,
            ) + 1
            out.append((nav_classes, nav_boxes, n_nav, (h_full, w_full)))
        return out

    def _upsample_groups(groups_1ghw: torch.Tensor, h: int, w: int) -> torch.Tensor:
        import torch.nn.functional as F

        x = groups_1ghw.unsqueeze(0)
        if x.dtype != torch.float16:
            x = x.to(dtype=torch.float16)
        return F.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=False
        ).clamp_(0.0, 1.0)[0]

    def _front_viz(
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
        if not dora_viz:
            return None
        pending: dict[str, np.ndarray] = {}
        # Upsample FiLM maps only when trav/overlay still needs them.
        need_film_full = refined_nav is None or viz_overlay
        probs_full = None
        if need_film_full and probs_lr is not None:
            probs_full = _upsample_groups(probs_lr, h, w).float()
        if viz_overlay:
            if labels_np.shape[0] != h or labels_np.shape[1] != w:
                labels_np = cv2.resize(
                    labels_np, (w, h), interpolation=cv2.INTER_NEAREST
                )
                conf_np = cv2.resize(
                    conf_np.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
                )
            color = labels_to_color(labels_np, layout.n_groups)
            pending["overlay"] = blend_overlay(rgb, color, conf_np, overlay_alpha)
        pack_src = refined_nav if refined_nav is not None else probs_full
        if pack_src is None:
            return pending or None
        rgb_np, nontrav = _pack_trav_rgb(pack_src)
        if viz_send_trav:
            pending["trav"] = rgb_np
        if sam_overlay and refined_nav is not None:
            # One D2H for blend weight (pack already synced rgb_np).
            conf = nontrav.detach().float().cpu().numpy()
            ov = blend_overlay(rgb, rgb_np, conf, overlay_alpha)
            if people_mask is not None:
                pm = people_mask.detach().float().cpu().numpy() >= 0.5
                ov = ov.copy()
                ov[pm, 2] = np.maximum(ov[pm, 2], 200)
            pending["overlay"] = ov
        elif not pending and not viz_send_trav:
            # Still emit trav if nothing else would be sent.
            pending["trav"] = rgb_np
        return pending or None

    def _run_matched(meta_rgb: dict, rgbs: list, feat_event) -> None:
        nonlocal last_viz_t, viz_fut
        n_cams = len(rgbs)
        h = int(meta_rgb["height"])
        w = int(meta_rgb["width"])

        hs = _open_features(feat_event, non_blocking=True)
        if hs.dim() == 3:
            hs = hs.unsqueeze(0)

        t0 = time.perf_counter()
        with torch.cuda.stream(stream_film):
            # Native FiLM res for labels/boxes; upsample only for viz/trav.
            probs_all = parts.decode_probs_batch(hs, dec, aggregate=aggregate)
        sam_input = sam_encoder.preprocess_batch(rgbs)
        with torch.cuda.stream(stream_sam):
            sam_feats_all = sam_encoder.encode_tensor(
                sam_input, synchronize=False
            )
        with torch.cuda.stream(stream_film):
            maxv, arg = probs_all.max(dim=1)
            labels_u8 = torch.where(
                maxv >= threshold,
                (arg + 1).to(torch.uint8),
                torch.zeros_like(arg, dtype=torch.uint8),
            )
            labels_np_all = labels_u8.cpu().numpy()
            conf_np_all = maxv.float().cpu().numpy()
        nav_reqs = _nav_reqs_from_labels_batch(labels_np_all, h, w)
        stream_sam.synchronize()
        t1 = time.perf_counter()
        if profile:
            print(f"FiLM||SAM encode time: {(t1 - t0) * 1000.0} ms", flush=True)

        t2 = time.perf_counter()
        sam_nav_all = sam_decoder.decode_multi(sam_feats_all, nav_reqs)
        t3 = time.perf_counter()
        if profile:
            tim = sam_decoder.last_timing
            print(
                f"SAM decode time: {(t3 - t2) * 1000.0} ms "
                f"(trt={tim.get('trt_ms', 0):.1f} upscale={tim.get('upscale_ms', 0):.1f} "
                f"boxes={tim.get('n_boxes', 0)} upscales={tim.get('n_upscales', 0)})",
                flush=True,
            )

        # Stamp e2e before any viz pack / Dora→Rerun send so viewer cost does not
        # inflate the pipeline number (and so backpressure is visible separately).
        e2e_ms = _record_e2e(meta_rgb)

        if not dora_viz or viz_pool is None:
            return

        now = time.perf_counter()
        if viz_min_interval_ms > 0 and (now - last_viz_t) * 1000.0 < viz_min_interval_ms:
            return
        # Never stall the GPU loop on a slow Rerun/Dora consumer.
        if viz_fut is not None and not viz_fut.done():
            return

        frame_ids = list(meta_rgb.get("frame_ids") or [])
        front_i = 0
        for i, fid in enumerate(frame_ids):
            if _cam_short_name(str(fid), i) == "front":
                front_i = i
                break
        refined_front = sam_nav_all[front_i]
        people_front = _union(refined_front, people_groups)
        labels_np = np.ascontiguousarray(labels_np_all[front_i])
        conf_np = np.ascontiguousarray(conf_np_all[front_i])
        rgb_front = np.ascontiguousarray(rgbs[front_i])
        # Minimal GPU sync: stage maps to CPU, pack/blend/send on the worker.
        maps_cpu = refined_front.detach().to(dtype=torch.float32, device="cpu")
        people_cpu = people_front.detach().to(dtype=torch.float32, device="cpu")
        probs_lr_cpu = (
            probs_all[front_i].detach().to(dtype=torch.float16, device="cpu")
            if (refined_front is None or viz_overlay)
            else None
        )
        cam_meta = dict(meta_rgb)
        cam_meta["n_cams"] = n_cams
        if frame_ids:
            cam_meta["frame_id"] = str(frame_ids[front_i])

        def _viz_job():
            pending = _front_viz(
                rgb_front,
                probs_lr_cpu,
                labels_np,
                conf_np,
                maps_cpu,
                people_cpu,
                h=h,
                w=w,
            )
            _send_dora_viz(pending, cam_meta, e2e_ms=e2e_ms)

        viz_fut = viz_pool.submit(_viz_job)
        last_viz_t = now
        if profile:
            print("viz staged async (drop-if-busy)", flush=True)

    def _drop_feat(slot: tuple | None) -> None:
        if slot is None:
            return
        try:
            _ = _open_features(slot[1])
            del _
        except Exception:  # noqa: BLE001
            pass

    for event in dora:
        if event is None or event.get("type") == "STOP":
            break
        if event.get("type") != "INPUT":
            continue

        eid = event["id"]
        meta = event["metadata"]

        def _stamp_scalar(v, default: int = 0) -> int:
            if isinstance(v, (list, tuple)):
                v = v[0] if v else default
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        stamp_key = (
            _stamp_scalar(meta.get("stamp_sec", meta.get("stamp_sec0", 0))),
            _stamp_scalar(meta.get("stamp_nanosec", meta.get("stamp_nanosec0", 0))),
        )

        if eid == "rgb":
            rgbs = unpack_rgb_batch(event["value"], meta)
            pending_rgb = (stamp_key, meta, rgbs)
        elif eid == "features":
            _drop_feat(pending_feat)
            pending_feat = (stamp_key, event)
        else:
            continue

        if (
            pending_rgb is not None
            and pending_feat is not None
            and pending_rgb[0] == pending_feat[0]
        ):
            _, meta_rgb, rgbs = pending_rgb
            _, feat_event = pending_feat
            pending_rgb = None
            pending_feat = None
            _run_matched(meta_rgb, rgbs, feat_event)

    if viz_pool is not None:
        if viz_fut is not None:
            try:
                viz_fut.result(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
        viz_pool.shutdown(wait=False)

    e2e_path = str(cfg.get("e2e_latency_json", "") or "").strip()
    if e2e_samples:
        st = _e2e_stats(e2e_samples)
        log(
            f"[e2e ms exit] n={st['n']} median={st['median_ms']:.1f} "
            f"mean={st['mean_ms']:.1f} p90={st['p90_ms']:.1f} "
            f"min={st['min_ms']:.1f} max={st['max_ms']:.1f}"
        )
        if e2e_path:
            with open(e2e_path, "w", encoding="utf-8") as f:
                json.dump(st, f, indent=2)
            log(f"e2e latency JSON -> {e2e_path}")


if __name__ == "__main__":
    main()
