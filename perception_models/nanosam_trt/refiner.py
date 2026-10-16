"""NanoSAM TensorRT helpers (batched engines only).

  * ``NanoSamEncoder`` — ResNet18 image encoder engine
  * ``NanoSamDecoder`` — batched MobileSAM mask decoder engine

``sam_image_size`` must match both engines (512 -> embed 32; 1024 -> embed 64).
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _profile() -> bool:
    return os.environ.get("PROFILE", "0") == "1"


def _load_image_encoder(path: str):
    from nanosam.utils.predictor import load_image_encoder_engine

    return load_image_encoder_engine(path)


def _load_mask_decoder(path: str):
    from nanosam.utils.predictor import load_mask_decoder_engine

    return load_mask_decoder_engine(path)


class NanoSamEncoder:
    """TensorRT image encoder. Requires dynamic or fixed batch matching N."""

    def __init__(
        self,
        image_encoder: str,
        image_encoder_size: int = 1024,
        device: str = "cuda",
        logger=None,
    ):
        self.dev = device
        self.image_encoder_size = int(image_encoder_size)
        self.feature_size = self.image_encoder_size // 16
        self.log = logger
        self.engine_path = str(image_encoder)
        if self.log:
            self.log.info(
                f"NanoSAM encoder engine={image_encoder} "
                f"size={self.image_encoder_size} feature={self.feature_size}"
            )
        self.engine = _load_image_encoder(image_encoder)
        self.engine_batch = self._read_engine_batch()
        self.last_timing = {"encoder_ms": 0.0}
        self._image_mean = torch.tensor(
            [123.675, 116.28, 103.53], dtype=torch.float32
        ).view(3, 1, 1)
        self._image_std = torch.tensor(
            [58.395, 57.12, 57.375], dtype=torch.float32
        ).view(3, 1, 1)
        # Reused pinned NCHW canvas for preprocess_batch → non_blocking H2D.
        self._pin_nchw: torch.Tensor | None = None
        self.warmup()

    def _read_engine_batch(self) -> int | None:
        """Fixed input batch, or None if dynamic / unknown."""
        eng = getattr(self.engine, "engine", None)
        if eng is None:
            return None
        try:
            shape = tuple(eng.get_tensor_shape("image"))
        except Exception:
            return None
        if not shape:
            return None
        b = int(shape[0])
        return b if b > 0 else None

    def _info(self, msg: str):
        if self.log:
            self.log.info(msg)

    def warmup(self):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.encode_batch([dummy])
        if self.engine_batch is None or self.engine_batch >= 3:
            _ = self.encode_batch([dummy, dummy, dummy])
            self._info("NanoSAM encoder warmup complete (N=1 + N=3)")
        else:
            self._info(
                f"NanoSAM encoder warmup complete (engine_batch={self.engine_batch})"
            )

    def preprocess(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """CPU resize/normalize; returns float32 CPU tensor [1,3,S,S]."""
        return self.preprocess_batch([rgb_uint8])

    def _ensure_pin_nchw(self, n: int, size: int) -> torch.Tensor:
        pin = self._pin_nchw
        if pin is None or pin.shape[0] < n or pin.shape[2] != size:
            # Allocate outside inference_mode so warmup (encode_batch) does not
            # leave a pinned buffer that rejects inplace ops on the hot path.
            with torch.inference_mode(False):
                pin = torch.empty(
                    (n, 3, size, size), dtype=torch.float32, pin_memory=True
                )
            self._pin_nchw = pin
        return pin[:n]

    def preprocess_batch(self, rgbs: list) -> torch.Tensor:
        """List of HxWx3 uint8 RGB → pinned float32 ``[N,3,S,S]`` (letterboxed)."""
        if not rgbs:
            raise ValueError("empty rgb batch")
        size = self.image_encoder_size
        n = len(rgbs)
        out = self._ensure_pin_nchw(n, size)
        out.zero_()
        mean = self._image_mean
        std = self._image_std
        for i, rgb in enumerate(rgbs):
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            if w >= h:
                rw, rh = size, max(1, int(size * h / w))
            else:
                rh, rw = size, max(1, int(size * w / h))
            resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(resized).permute(2, 0, 1).to(dtype=torch.float32)
            x.sub_(mean).div_(std)
            out[i, :, :rh, :rw].copy_(x)
        return out

    @torch.inference_mode()
    def encode_tensor(
        self, image_tensor: torch.Tensor, *, synchronize: bool = True
    ) -> torch.Tensor:
        """Run TRT image encoder. ``image_tensor``: ``[N,3,S,S]`` (batched)."""
        if synchronize:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.to(
                device=self.dev, dtype=torch.float32, non_blocking=True
            )
        n = int(image_tensor.shape[0])
        eng_b = self.engine_batch
        if eng_b is not None and eng_b != n:
            raise RuntimeError(
                f"NanoSAM encoder engine batch={eng_b} but got N={n}; "
                "use a dynamic/_bN engine (no serial fallback)"
            )
        features = self.engine(image_tensor)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if synchronize:
            torch.cuda.synchronize()
            self.last_timing = {"encoder_ms": (time.perf_counter() - t0) * 1000.0}
        return features

    @torch.inference_mode()
    def encode(self, rgb_uint8: np.ndarray, *, synchronize: bool = True) -> torch.Tensor:
        return self.encode_tensor(self.preprocess(rgb_uint8), synchronize=synchronize)

    @torch.inference_mode()
    def encode_batch(
        self, rgbs: list, *, synchronize: bool = True
    ) -> torch.Tensor:
        """Encode N RGBs → features ``[N,C,Hf,Wf]`` (one TRT enqueue)."""
        return self.encode_tensor(self.preprocess_batch(rgbs), synchronize=synchronize)


class NanoSamDecoder:
    """TensorRT mask decoder — batched prompts, reusable CUDA buffers."""

    def __init__(
        self,
        mask_decoder: str,
        orig_image_encoder_size: int = 1024,
        device: str = "cuda",
        logger=None,
        max_batch_k: int = 16,
    ):
        self.dev = device
        self.orig_image_encoder_size = int(orig_image_encoder_size)
        self.feature_size = self.orig_image_encoder_size // 16
        self.mask_input_size = 4 * self.feature_size
        self.max_batch_k = max(1, int(max_batch_k))
        self.log = logger
        if self.log:
            self.log.info(
                f"NanoSAM decoder engine={mask_decoder} "
                f"canvas={self.orig_image_encoder_size} "
                f"feature={self.feature_size} mask_in={self.mask_input_size} "
                f"max_batch_k={self.max_batch_k}"
            )
        self.engine = _load_mask_decoder(mask_decoder)
        ms = self.mask_input_size
        k = self.max_batch_k
        self._point_coords = torch.zeros(k, 2, 2, device=self.dev, dtype=torch.float32)
        self._point_labels = torch.zeros(k, 2, device=self.dev, dtype=torch.float32)
        self._point_labels[:, 0] = 2.0
        self._point_labels[:, 1] = 3.0
        self._mask_input = torch.zeros(k, 1, ms, ms, device=self.dev, dtype=torch.float32)
        self._has_mask_input = torch.zeros(k, device=self.dev, dtype=torch.float32)
        self._pts_host = np.zeros((k, 2, 2), dtype=np.float32)
        self.last_timing = {
            "decoder_ms": 0.0,
            "trt_ms": 0.0,
            "upscale_ms": 0.0,
            "n_boxes": 0,
            "n_upscales": 0,
            "decoder_ms_per_box": 0.0,
            "decoder_mode": "batched",
        }
        self.warmup()

    def _warn(self, msg):
        if self.log:
            self.log.warn(msg)

    def _info(self, msg):
        if self.log:
            self.log.info(msg)

    def _ensure_k(self, k: int) -> None:
        if k <= self.max_batch_k:
            return
        # Grow buffers if a frame exceeds the warm capacity (rare).
        self.max_batch_k = int(k)
        ms = self.mask_input_size
        self._point_coords = torch.zeros(
            k, 2, 2, device=self.dev, dtype=torch.float32
        )
        self._point_labels = torch.zeros(k, 2, device=self.dev, dtype=torch.float32)
        self._point_labels[:, 0] = 2.0
        self._point_labels[:, 1] = 3.0
        self._mask_input = torch.zeros(
            k, 1, ms, ms, device=self.dev, dtype=torch.float32
        )
        self._has_mask_input = torch.zeros(k, device=self.dev, dtype=torch.float32)
        self._pts_host = np.zeros((k, 2, 2), dtype=np.float32)
        self._info(f"NanoSAM decoder buffers grew to max_batch_k={k}")

    def warmup(self):
        fs = self.feature_size
        feat = torch.zeros(1, 256, fs, fs, device=self.dev, dtype=torch.float32)
        boxes = [
            np.array([80, 80, 200, 200], dtype=np.float32),
            np.array([220, 120, 400, 360], dtype=np.float32),
        ]
        try:
            _ = self._predict_boxes_batched(feat, boxes, (480, 640))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"NanoSAM mask decoder batch unsupported ({e}). Rebuild with "
                "perception_models/scripts/export_nanosam_mask_decoder_batched_onnx.py + "
                "perception_models/scripts/build_nanosam_mask_decoder_batched_engine.sh"
            ) from e
        self._info("NanoSAM decoder warmup complete (mode=batched).")

    def _scale_points_inplace(self, pts_host: np.ndarray, image_hw, k: int) -> None:
        """Scale first k box-corners in ``pts_host`` into encoder canvas coords."""
        h, w = int(image_hw[0]), int(image_hw[1])
        scale = self.orig_image_encoder_size / float(max(h, w))
        pts_host[:k] *= scale

    def _upscale_low_res(self, low_res: torch.Tensor, image_hw) -> torch.Tensor:
        """Letterbox crop of low-res logits → full ``image_hw`` (batched)."""
        size = self.mask_input_size
        ih, iw = int(image_hw[0]), int(image_hw[1])
        if iw > ih:
            lim_x = size
            lim_y = max(1, int(size * ih / iw))
        else:
            lim_y = size
            lim_x = max(1, int(size * iw / ih))
        return F.interpolate(
            low_res[:, :, :lim_y, :lim_x],
            size=(ih, iw),
            mode="bilinear",
            align_corners=False,
        )

    def _predict_boxes_batched(
        self, features: torch.Tensor, boxes_xyxy, image_hw
    ) -> torch.Tensor:
        """``features`` ``[1,C,H,W]`` → expand to one row per box, then decode."""
        k = len(boxes_xyxy)
        if k == 0:
            h, w = int(image_hw[0]), int(image_hw[1])
            return torch.zeros((0, h, w), device=self.dev, dtype=torch.float32)
        if features.dim() == 3:
            features = features.unsqueeze(0)
        features_b = features.expand(k, -1, -1, -1).contiguous()
        low_res, _ = self._run_decoder(features_b, boxes_xyxy, image_hw)
        hi = self._upscale_low_res(low_res, image_hw)
        return (hi[:, 0] > 0).float()

    def _run_decoder(
        self, features_b: torch.Tensor, boxes_xyxy, image_hw
    ) -> tuple[torch.Tensor, dict]:
        """TRT decode → low-res logits ``[K,1,ms,ms]``. Reuses CUDA buffers."""
        k = len(boxes_xyxy)
        if k == 0:
            ms = self.mask_input_size
            empty = torch.zeros(0, 1, ms, ms, device=self.dev, dtype=torch.float32)
            return empty, {"trt_ms": 0.0}
        if int(features_b.shape[0]) != k:
            raise ValueError(
                f"features_b batch {features_b.shape[0]} != n_boxes {k}"
            )
        self._ensure_k(k)

        pts = self._pts_host
        for i, box in enumerate(boxes_xyxy):
            x0, y0, x1, y1 = [float(v) for v in box]
            pts[i, 0, 0] = x0
            pts[i, 0, 1] = y0
            pts[i, 1, 0] = x1
            pts[i, 1, 1] = y1
        self._scale_points_inplace(pts, image_hw, k)

        self._point_coords[:k].copy_(
            torch.from_numpy(pts[:k]), non_blocking=True
        )
        # Labels stay 2/3; mask_input / has_mask stay zero (no dense mask prompt).

        profile = _profile()
        if profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        _iou, low_res = self.engine(
            features_b.contiguous(),
            self._point_coords[:k],
            self._point_labels[:k],
            self._mask_input[:k],
            self._has_mask_input[:k],
        )
        if low_res.shape[1] > 1:
            low_res = low_res[:, :1]

        if profile:
            torch.cuda.synchronize()
            trt_ms = (time.perf_counter() - t0) * 1000.0
        else:
            trt_ms = 0.0
        return low_res, {"trt_ms": trt_ms}

    @torch.inference_mode()
    def decode(
        self,
        features: torch.Tensor,
        classes,
        boxes_xyxy,
        n_classes: int,
        image_hw,
    ) -> torch.Tensor:
        """features: [1,C,H,W]. Returns [n_classes,H,W] float 0/1 on CUDA."""
        outs = self.decode_multi(
            features if features.dim() == 4 else features.unsqueeze(0),
            [(classes, boxes_xyxy, n_classes, image_hw)],
        )
        return outs[0]

    @torch.inference_mode()
    def decode_multi(
        self,
        features: torch.Tensor,
        requests: list,
    ) -> list:
        """Batched mask decode across images (and their boxes).

        ``features``: ``[N,C,Hf,Wf]``
        ``requests``: list length N of
          ``(classes, boxes_xyxy, n_classes, image_hw)``
        Returns list of ``[n_classes,H,W]`` tensors.

        Upscales once per unique ``(image, class)`` after OR-ing low-res logits
        (cheaper than upscaling every box).
        """
        if features.dim() == 3:
            features = features.unsqueeze(0)
        n_img = int(features.shape[0])
        if len(requests) != n_img:
            raise ValueError(
                f"decode_multi: features N={n_img} but {len(requests)} requests"
            )

        outs = []
        img_ids: list[int] = []
        boxes_all = []
        meta = []  # (img_i, cls)
        image_hw_ref = None

        for i, req in enumerate(requests):
            classes, boxes_xyxy, n_classes, image_hw = req
            h, w = int(image_hw[0]), int(image_hw[1])
            if image_hw_ref is None:
                image_hw_ref = (h, w)
            elif image_hw_ref != (h, w):
                raise ValueError(
                    f"decode_multi requires uniform image_hw; "
                    f"got {image_hw_ref} and {(h, w)}"
                )
            refined = torch.zeros(
                (int(n_classes), h, w), device=self.dev, dtype=torch.float32
            )
            outs.append(refined)
            if not boxes_xyxy:
                continue
            for cls, box in zip(classes, boxes_xyxy):
                if cls < 0 or cls >= n_classes:
                    continue
                x0, y0, x1, y1 = [float(v) for v in box]
                x0 = max(0, min(w - 1, x0))
                x1 = max(0, min(w - 1, x1))
                y0 = max(0, min(h - 1, y0))
                y1 = max(0, min(h - 1, y1))
                if x1 <= x0 or y1 <= y0:
                    continue
                img_ids.append(i)
                boxes_all.append(np.array([x0, y0, x1, y1], dtype=np.float32))
                meta.append((i, int(cls)))

        self.last_timing = {
            "decoder_ms": 0.0,
            "trt_ms": 0.0,
            "upscale_ms": 0.0,
            "n_boxes": 0,
            "n_upscales": 0,
            "decoder_ms_per_box": 0.0,
            "decoder_mode": "batched",
            "n_images": n_img,
        }
        if not boxes_all:
            return outs

        profile = _profile()
        if profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        idx = torch.tensor(img_ids, device=features.device, dtype=torch.long)
        features_b = features.index_select(0, idx).contiguous()
        low_res, trt_info = self._run_decoder(features_b, boxes_all, image_hw_ref)

        # OR low-res logits per (image, class), then one upsample each.
        merged: dict[tuple[int, int], torch.Tensor] = {}
        for mi, key in enumerate(meta):
            row = low_res[mi : mi + 1]
            prev = merged.get(key)
            merged[key] = row if prev is None else torch.maximum(prev, row)

        if profile:
            torch.cuda.synchronize()
            t_up0 = time.perf_counter()

        for (img_i, cls), lr in merged.items():
            hi = self._upscale_low_res(lr, image_hw_ref)
            outs[img_i][cls] = (hi[0, 0] > 0).float()

        if profile:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            decoder_ms = (t1 - t0) * 1000.0
            upscale_ms = (t1 - t_up0) * 1000.0
            trt_ms = float(trt_info.get("trt_ms", 0.0))
        else:
            decoder_ms = 0.0
            upscale_ms = 0.0
            trt_ms = 0.0

        n_pred = len(boxes_all)
        self.last_timing = {
            "decoder_ms": decoder_ms,
            "trt_ms": trt_ms,
            "upscale_ms": upscale_ms,
            "n_boxes": n_pred,
            "n_upscales": len(merged),
            "decoder_ms_per_box": (decoder_ms / n_pred) if n_pred else 0.0,
            "decoder_mode": "batched",
            "n_images": n_img,
        }
        return outs
