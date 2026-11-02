#!/usr/bin/env python3
"""Dora-only prompted segmentation: box proposals ∥ NanoSAM encode, then batched SAM decode.

Two sources of box proposals (``object_source``):

* ``clipseg`` (default): CLIPSeg FiLM maps in this process (below).
* ``yolo_world``: boxes from the ``dora_yolo_world`` detector node. The camera
  frames come straight into this node too. NanoSAM encodes each frame as soon
  as it arrives, while YOLO-World detects the same frame in its own process.
  When the detections with the matching stamp arrive, their boxes go through the
  same single batched decode. Prompts then go to the detector, not here.
  CLIPSeg is not loaded in this mode; it is meant for scene semantics.

CLIPSeg path, per frame (N cameras, one batch):

1. FiLM decode for all ``N x P`` image-prompt pairs runs on ``stream_film``.
   At the same time, the NanoSAM image encoder runs for all N images on
   ``stream_sam``. The two have independent inputs, so they overlap.
2. Each prompt is thresholded independently. Connected components become
   instance boxes (``instance_select``).
3. One batched NanoSAM mask-decoder enqueue covers every box of every camera.

Prompts change at runtime through the ``prompt`` input. Tokenization and the
CLIP text tower run on a background thread (:class:`PromptBank`), and the
frame loop picks up the new ``cond`` snapshot atomically on its next frame.

Outputs:

* ``masks``: self-describing ``seg_masks`` message (see ``dora_io.encode_seg_masks``).
  It holds per-camera instance masks for the prompted object(s) only, with
  stamps copied from the source camera frames.
* ``overlay``: optional RGB overlay of the leader camera, if ``dora_viz`` is set.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import torch

from dora import Node

from perception_models.clipseg.shared import (
    ClipSegParts,
    PromptLayout,
    load_seg_config,
    prompt_layout_from_config,
)
from perception_models.clipseg.trt import TrtFilmDecoder, film_engine_error, load_clipseg_film_engine
from perception_models.cuda_ipc import unpack_arrow_f16, unpack_cuda_tensor
from perception_models.dora_io import (
    LatestFrameSync,
    WireError,
    decode,
    decode_camera_frame,
    encode_seg_masks,
    is_camera_input,
    join_stamp,
    parse_prompts,
)
from perception_models.instance_select import Instance, instances_from_detections, propose_instances
from perception_models.multicam_rgb_batch import unpack_rgb_batch
from perception_models.nanosam_trt import NanoSamDecoder, NanoSamEncoder
from perception_models.prompt_bank import PromptBank, PromptSet
from perception_models.seg_ops import PALETTE, blend_overlay


def _feature_shape(meta: dict) -> tuple[int, ...]:
    n_batch = int(meta.get("feature_batch", meta.get("n_cams", 1)))
    base = (int(meta["n_layers"]), int(meta["seq_len"]), int(meta["hidden"]))
    return (n_batch, *base) if n_batch > 1 else base


def _open_features(event, *, non_blocking: bool = False) -> torch.Tensor:
    meta = event["metadata"]
    mode = str(meta.get("feature_mode", "arrow_f16"))
    shape = _feature_shape(meta)
    if mode == "torch_cuda_ipc":
        return unpack_cuda_tensor(event["value"], meta).reshape(*shape).contiguous()
    return unpack_arrow_f16(event["value"], shape, non_blocking=non_blocking)


def _stamp_scalar(v, default: int = 0) -> int:
    if isinstance(v, (list, tuple)):
        v = v[0] if v else default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def frame_stamps(meta: dict, n: int) -> list[float]:
    """Float stamps per camera: exact ``stamp_list`` if present, else rebuilt from sec/nsec."""
    exact = meta.get("stamp_list")
    if exact is not None and len(exact) >= n:
        return [float(s) for s in exact[:n]]
    secs = list(meta.get("stamp_sec_list") or [meta.get("stamp_sec", 0)] * n)
    nsecs = list(meta.get("stamp_nanosec_list") or [meta.get("stamp_nanosec", 0)] * n)
    return [join_stamp(int(secs[min(i, len(secs) - 1)]), int(nsecs[min(i, len(nsecs) - 1)])) for i in range(n)]


def layout_for(prompts: tuple[str, ...]) -> PromptLayout:
    """One group per prompt; no suppress prompts (we only want the prompted thing)."""
    return PromptLayout(subprompts=list(prompts), group_names=[], group_ids=[], suppress_prompts=[])


class SegRefineNode:
    """Box proposals (CLIPSeg FiLM or YOLO-World) ∥ NanoSAM encode → batched decode → instance masks."""

    def __init__(self, cfg: dict):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; seg_refine needs a GPU")
        torch.set_float32_matmul_precision("high")

        self.cfg = cfg
        self.dora = Node()
        self.profile = os.environ.get("PROFILE", "0") == "1" or bool(cfg.get("profile", False))
        if self.profile:
            os.environ["PROFILE"] = "1"

        self.batch_n = max(1, int(cfg.get("clipseg_batch_n", 1)))
        self.object_source = str(cfg.get("object_source", "clipseg")).lower()
        if self.object_source not in ("clipseg", "yolo_world"):
            raise ValueError(f"object_source must be clipseg or yolo_world, got {self.object_source!r}")
        self.max_inst = int(cfg.get("max_instances_per_prompt", 2))
        self.min_area_frac = float(cfg.get("sam_min_area", 0.0005))

        enc_path = str(cfg.get("sam_image_encoder", "") or "").strip()
        dec_path = str(cfg.get("sam_mask_decoder", "") or "").strip()
        if not enc_path or not dec_path:
            raise RuntimeError("sam_image_encoder and sam_mask_decoder required (batched TRT)")
        sam_size = int(cfg.get("sam_image_size", 512))
        self.log("Loading NanoSAM (TensorRT) ...")
        self.sam_encoder = NanoSamEncoder(enc_path, image_encoder_size=sam_size, device="cuda", logger=_Logger())
        self.sam_decoder = NanoSamDecoder(
            dec_path,
            orig_image_encoder_size=sam_size,
            device="cuda",
            logger=_Logger(),
            max_batch_k=max(4, int(cfg.get("sam_decoder_max_batch", 16))),
        )
        self.stream_sam = torch.cuda.Stream()
        self.prompts: PromptBank | None = None
        if self.object_source == "clipseg":
            self._init_clipseg(cfg)
        else:
            self._init_yolo(cfg)

        self.pending_rgb: tuple | None = None
        self.pending_feat: tuple | None = None

        self.dora_viz = bool(cfg.get("dora_viz", False))
        self.overlay_alpha = float(cfg.get("overlay_alpha", 0.5))
        self.viz_pool = ThreadPoolExecutor(max_workers=1) if self.dora_viz else None
        self.viz_fut = None
        self.box_pool = ThreadPoolExecutor(max_workers=4)

        self.e2e_samples: list[float] = []
        self.e2e_log_every_s = max(1.0, float(cfg.get("e2e_log_every_s", 30.0)))
        self.e2e_log_t0 = time.perf_counter()
        self.log(
            f"dora seg_refine up: {self.object_source} boxes ∥ SAM-encode → batched decode; "
            f"batch_n={self.batch_n} {self.film_info} viz={'on' if self.dora_viz else 'off'}"
        )

    @staticmethod
    def log(msg: str) -> None:
        print(f"[dora_seg_refine] {msg}", flush=True)

    def _init_clipseg(self, cfg: dict) -> None:
        initial_layout = prompt_layout_from_config(cfg)
        self.parts = ClipSegParts(
            str(cfg.get("model_name", "CIDAS/clipseg-rd16")),
            initial_layout,
            input_res=int(cfg.get("input_res", 224)),
            compile_mode="none",
            device="cuda",
            logger=self.log,
        )
        self.dec = self._load_film_decoder()
        self.aggregate = str(cfg.get("aggregate", "max")).lower()
        self.threshold = float(cfg.get("threshold", 0.4))
        self.box_pad = float(cfg.get("box_pad_frac", 0.08))
        self.stream_film = torch.cuda.Stream()
        self.stream_text = torch.cuda.Stream()

        # Prompt embeddings: startup set embedded synchronously, then live updates in the background.
        self._layouts: dict[int, PromptLayout] = {}
        self.prompts = PromptBank(
            lambda ps: self.parts.embed_prompts(ps, stream=self.stream_text),
            max_prompts=self.film_slots,
            on_update=lambda s: self.log(f"prompts v{s.version} active: {list(s.prompts)}"),
            on_error=lambda ps, e: self.log(f"WARN prompt embedding failed for {list(ps)}: {e}"),
        )
        start = [p for p in initial_layout.subprompts if p]
        if start:
            self.prompts.set_blocking(start)
        self.film_info = f"prompt_slots={self.film_slots}"

    def _init_yolo(self, cfg: dict) -> None:
        self.det_min_score = float(cfg.get("det_min_score", 0.0))
        self.box_pad = float(cfg.get("det_box_pad_frac", 0.02))
        cams = [str(c) for c in (cfg.get("camera_inputs") or []) if str(c)]
        self.camera_inputs = cams
        self._sync: LatestFrameSync | None = LatestFrameSync(cams) if cams else None
        # Frames SAM-encoded but not yet decoded, and detections that beat their frame here.
        # Keyed by the leader camera's float stamp, which both nodes copy from the same frame.
        depth = max(2, int(cfg.get("match_depth", 4)))
        self._encoded: deque[tuple[float, dict]] = deque(maxlen=depth)
        self._early_dets: deque[tuple[float, dict]] = deque(maxlen=depth)
        self.n_unmatched = 0
        # Warm the encoder so the first real frame is not slowed by lazy init.
        dummy = [np.zeros((64, 64, 3), np.uint8)] * self.batch_n
        _ = self.sam_encoder.encode_tensor(self.sam_encoder.preprocess_batch(dummy), synchronize=True)
        self.film_info = f"detector boxes (camera_inputs={cams or 'auto'})"

    def _load_film_decoder(self):
        """FiLM decoder: TRT engine if it matches eager on a self-check, else eager fp16.

        ``film_backend``: ``auto`` (default), ``trt`` (fail if the check fails) or
        ``torch``. Either way the decoder is called with ``(stacked_hs, cond)`` and
        returns ``[N, P, H, W]``.
        """
        backend = str(self.cfg.get("film_backend", "auto")).lower()
        eager = self.parts.build_decoder()
        hs_shape = [self.batch_n, len(self.parts.extract), (self.parts.res // 16) ** 2 + 1, 768]
        if backend != "torch":
            path = str(self.cfg.get("clipseg_film_engine", "") or "").strip()
            if path and os.path.isfile(path):
                film_eng = load_clipseg_film_engine(path)
                dec = TrtFilmDecoder(film_eng)
                try:
                    err = film_engine_error(
                        dec, eager, hs_shape, int(self.parts.cond.shape[1]), int(dec._num_prompts or 1)
                    )
                except Exception as e:  # noqa: BLE001 - e.g. engine built for another input_res
                    self.log(f"FiLM TRT engine unusable: {e}")
                    err = float("inf")
                tol = float(self.cfg.get("film_check_tol", 0.05))
                if err <= tol:
                    self.log(f"CLIPSeg FiLM: TensorRT {path} (self-check err {err:.4f})")
                    self.film_slots = dec._num_prompts
                    return dec
                msg = f"FiLM TRT engine disagrees with eager (max err {err:.3f} > {tol}); "
                if backend == "trt":
                    raise RuntimeError(msg + "rebuild the engine or set film_backend: torch")
                self.log("WARNING " + msg + "using eager fp16 FiLM")
            elif backend == "trt":
                raise RuntimeError(f"clipseg_film_engine not found: {path!r}")
        self.film_slots = int(self.cfg.get("prompt_slots", 8))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(3):
                _ = eager(torch.zeros(*hs_shape, device="cuda"), self.parts.cond)
        torch.cuda.synchronize()
        self.log("CLIPSeg FiLM: eager fp16 (warm)")
        return eager

    def _layout(self, snap: PromptSet) -> PromptLayout:
        lay = self._layouts.get(snap.version)
        if lay is None:
            self._layouts = {snap.version: layout_for(snap.prompts)}
            lay = self._layouts[snap.version]
        return lay

    # ------------------------------------------------------------ hot path
    def _run_matched(self, meta_rgb: dict, rgbs: list, feat_event) -> None:
        assert self.prompts is not None
        snap = self.prompts.current()
        hs = _open_features(feat_event, non_blocking=True)
        if snap is None:
            return  # no prompt yet
        if hs.dim() == 3:
            hs = hs.unsqueeze(0)
        n_cams = len(rgbs)
        h, w = int(meta_rgb["height"]), int(meta_rgb["width"])
        layout = self._layout(snap)

        t0 = time.perf_counter()
        # Features were copied H2D (non_blocking) on the default stream.
        self.stream_film.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream_film):
            probs_all = self.parts.decode_probs_batch(
                hs, self.dec, aggregate=self.aggregate, cond=snap.cond, layout=layout
            )
        sam_input = self.sam_encoder.preprocess_batch(rgbs)
        with torch.cuda.stream(self.stream_sam):
            sam_feats_all = self.sam_encoder.encode_tensor(sam_input, synchronize=False)
        with torch.cuda.stream(self.stream_film):
            probs_np = probs_all.float().cpu().numpy()  # [N,P,h,w]; syncs stream_film only

        def _props(i: int) -> list[Instance]:
            return propose_instances(
                probs_np[i],
                (h, w),
                threshold=self.threshold,
                min_area_frac=self.min_area_frac,
                max_instances_per_prompt=self.max_inst,
                box_pad_frac=self.box_pad,
            )

        insts = list(self.box_pool.map(_props, range(n_cams))) if n_cams > 1 else [_props(0)]
        self.stream_sam.synchronize()
        t1 = time.perf_counter()

        # Each instance is its own "class", so decode_multi returns one mask per box.
        reqs = [(list(range(len(il))), [i.box for i in il], max(1, len(il)), (h, w)) for il in insts]
        masks_all = self.sam_decoder.decode_multi(sam_feats_all, reqs)
        t2 = time.perf_counter()
        if self.profile:
            tim = self.sam_decoder.last_timing
            print(
                f"FiLM||SAM-enc {(t1 - t0) * 1e3:.1f} ms | SAM decode {(t2 - t1) * 1e3:.1f} ms "
                f"(boxes={tim.get('n_boxes', 0)})",
                flush=True,
            )

        stamps = frame_stamps(meta_rgb, n_cams)
        names = [str(x) for x in (meta_rgb.get("frame_ids") or [])] + [f"cam{i}" for i in range(n_cams)]
        self._emit(stamps, names[:n_cams], (h, w), insts, masks_all, list(snap.prompts), snap.version, meta_rgb, rgbs[0])

    def _emit(self, stamps, names, hw, insts, masks_all, prompts, version, meta_src, rgb0) -> None:
        """Pack per-camera instance masks into one ``seg_masks`` message and send it."""
        h, w = hw
        cameras = []
        for i in range(len(insts)):
            k = len(insts[i])
            m = (masks_all[i][:k] > 0.5).to(torch.uint8).cpu().numpy() if k else np.zeros((0, h, w), np.uint8)
            cameras.append(
                {
                    "name": names[i],
                    "stamp": stamps[i],
                    "height": h,
                    "width": w,
                    "masks": m,
                    "prompt_idx": np.array([x.prompt_idx for x in insts[i]], dtype=np.int32),
                    "scores": np.array([x.score for x in insts[i]], dtype=np.float32),
                    "boxes": np.array([x.box for x in insts[i]], dtype=np.float32).reshape(k, 4),
                }
            )
        out_meta = {"stamp": stamps[0], "prompt_version": int(version), "n_cams": len(insts)}
        self.dora.send_output(
            "masks",
            encode_seg_masks(stamp=stamps[0], prompts=list(prompts), prompt_version=int(version), cameras=cameras),
            out_meta,
        )
        self._record_e2e(meta_src)
        self._maybe_viz(rgb0, cameras[0], out_meta)

    # ------------------------------------------------- hot path (yolo_world)
    def _on_yolo_camera(self, input_id: str, event) -> None:
        """New camera frame: start its SAM encode now, overlapping the detector's work on it."""
        try:
            frame = decode_camera_frame(event["value"])
        except WireError as e:
            self.log(f"WARN {input_id}: {e}")
            return
        frame["_t_rx"] = time.perf_counter()
        if self._sync is None:
            self._sync = LatestFrameSync([input_id])
        if input_id not in self._sync.cameras:
            if len(self._sync.cameras) >= self.batch_n:
                return
            self._sync.cameras.append(input_id)
        batch = self._sync.push(input_id, frame)
        if batch is None:
            return
        rgbs = [f["rgb"] for f in batch]
        if len({r.shape for r in rgbs}) != 1:
            self.log(f"WARN skip: mismatched RGB shapes {[r.shape for r in rgbs]}")
            return
        sam_input = self.sam_encoder.preprocess_batch(rgbs)
        self.stream_sam.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream_sam):
            feats = self.sam_encoder.encode_tensor(sam_input, synchronize=False)
            done = torch.cuda.Event()
            done.record(self.stream_sam)
        key = float(batch[0]["stamp"])
        entry = {
            "rgbs": rgbs,
            "feats": feats,
            "done": done,
            "stamps": [float(f["stamp"]) for f in batch],
            "names": [str(f.get("name") or c) for f, c in zip(batch, self._sync.cameras)],
            "t_src_perf": batch[0]["_t_rx"],
        }
        for i, (k, det) in enumerate(self._early_dets):
            if k == key:
                del self._early_dets[i]
                self._decode_yolo(entry, det)
                return
        if len(self._encoded) == self._encoded.maxlen:
            self.n_unmatched += 1  # oldest frame never got detections (detector dropped it)
        self._encoded.append((key, entry))

    def _on_detections(self, event) -> None:
        try:
            kind, det = decode(event["value"])
        except Exception as e:  # noqa: BLE001
            self.log(f"WARN bad detections: {e}")
            return
        if kind != "detections":
            self.log(f"WARN expected a detections message, got {kind!r}")
            return
        key = float(det["stamp"])
        for i, (k, entry) in enumerate(self._encoded):
            if k == key:
                # Older frames can no longer match: detections arrive in frame order.
                for _ in range(i + 1):
                    self._encoded.popleft()
                self._decode_yolo(entry, det)
                return
        self._early_dets.append((key, det))

    def _decode_yolo(self, entry: dict, det: dict) -> None:
        t0 = time.perf_counter()
        cams = list(det.get("cameras") or [])
        h, w = entry["rgbs"][0].shape[:2]
        insts: list[list[Instance]] = []
        for i in range(len(entry["rgbs"])):
            c = cams[i] if i < len(cams) else {}
            insts.append(
                instances_from_detections(
                    np.asarray(c.get("boxes", np.zeros((0, 4))), dtype=np.float64),
                    np.asarray(c.get("scores", []), dtype=np.float64),
                    np.asarray(c.get("class_idx", []), dtype=np.int64),
                    (h, w),
                    min_score=self.det_min_score,
                    max_instances_per_prompt=self.max_inst,
                    box_pad_frac=self.box_pad,
                )
            )
        entry["done"].synchronize()
        t1 = time.perf_counter()
        reqs = [(list(range(len(il))), [x.box for x in il], max(1, len(il)), (h, w)) for il in insts]
        masks_all = self.sam_decoder.decode_multi(entry["feats"], reqs)
        if self.profile:
            print(
                f"SAM-enc wait {(t1 - t0) * 1e3:.1f} ms | SAM decode {(time.perf_counter() - t1) * 1e3:.1f} ms "
                f"(boxes={sum(len(il) for il in insts)})",
                flush=True,
            )
        self._emit(
            entry["stamps"],
            entry["names"],
            (h, w),
            insts,
            masks_all,
            list(det.get("prompts") or []),
            int(det.get("prompt_version", 0)),
            {"t_src_perf": entry["t_src_perf"]},
            entry["rgbs"][0],
        )

    def _record_e2e(self, meta_rgb: dict) -> None:
        t0 = meta_rgb.get("t_src_perf")
        if t0 is None:
            return
        self.e2e_samples.append((time.perf_counter() - float(t0)) * 1000.0)
        now = time.perf_counter()
        if now - self.e2e_log_t0 >= self.e2e_log_every_s and self.e2e_samples:
            arr = np.asarray(self.e2e_samples[-500:])
            self.log(f"[e2e ms] n={arr.size} median={np.median(arr):.1f} p90={np.percentile(arr, 90):.1f}")
            self.e2e_log_t0 = now
            path = str(self.cfg.get("e2e_latency_json", "") or "")
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"median_ms": float(np.median(arr)), "n": int(arr.size)}, f)

    def _maybe_viz(self, rgb: np.ndarray, cam: dict, meta: dict) -> None:
        if self.viz_pool is None or (self.viz_fut is not None and not self.viz_fut.done()):
            return

        def job():
            color = np.zeros_like(rgb)
            conf = np.zeros(rgb.shape[:2], np.float32)
            for j, m in enumerate(cam["masks"]):
                color[m > 0] = PALETTE[j % len(PALETTE)]
                conf[m > 0] = 1.0
            ov = np.ascontiguousarray(blend_overlay(rgb, color, conf, self.overlay_alpha))
            self.dora.send_output(
                "overlay",
                pa.array(ov.reshape(-1), type=pa.uint8()),
                {**meta, "height": int(ov.shape[0]), "width": int(ov.shape[1]), "encoding": "rgb8"},
            )

        self.viz_fut = self.viz_pool.submit(job)

    # -------------------------------------------------------------- events
    def _on_event(self, event) -> None:
        eid = event["id"]
        if self.object_source == "yolo_world":
            if eid == "detections":
                self._on_detections(event)
            elif eid == "prompt":
                self.log("WARN prompt input ignored: with object_source=yolo_world, send prompts to the detector")
            elif is_camera_input(eid, self.camera_inputs):
                self._on_yolo_camera(eid, event)
            return
        if eid == "prompt":
            try:
                prompts = parse_prompts(event["value"])
                if prompts and self.prompts is not None and self.prompts.request(prompts):
                    self.log(f"prompt update queued: {prompts}")
            except Exception as e:  # noqa: BLE001
                self.log(f"WARN bad prompt update: {e}")
            return
        meta = event["metadata"]
        stamp_key = (_stamp_scalar(meta.get("stamp_sec", 0)), _stamp_scalar(meta.get("stamp_nanosec", 0)))
        if eid == "rgb":
            self.pending_rgb = (stamp_key, meta, unpack_rgb_batch(event["value"], meta))
        elif eid == "features":
            self.pending_feat = (stamp_key, event)
        else:
            return
        if self.pending_rgb and self.pending_feat and self.pending_rgb[0] == self.pending_feat[0]:
            _, meta_rgb, rgbs = self.pending_rgb
            _, feat_event = self.pending_feat
            self.pending_rgb = self.pending_feat = None
            self._run_matched(meta_rgb, rgbs, feat_event)

    def run(self) -> None:
        try:
            for event in self.dora:
                if event is None or event.get("type") == "STOP":
                    break
                if event.get("type") == "INPUT":
                    self._on_event(event)
        finally:
            if self.prompts is not None:
                self.prompts.close()
            self.box_pool.shutdown(wait=False)
            if self.viz_pool is not None:
                self.viz_pool.shutdown(wait=False)


class _Logger:
    def info(self, msg: str) -> None:
        SegRefineNode.log(msg)

    def warning(self, msg: str) -> None:
        SegRefineNode.log(f"WARNING: {msg}")

    warn = warning


def main():
    SegRefineNode(load_seg_config()).run()


if __name__ == "__main__":
    main()
