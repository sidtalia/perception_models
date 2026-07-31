"""NanoSAM TensorRT helpers.

  * ``NanoSamEncoder`` — ResNet18 image encoder engine
  * ``NanoSamDecoder`` — batched MobileSAM mask decoder engine

``sam_image_size`` must match both engines (512 -> embed 32; 1024 -> embed 64).
"""

from __future__ import annotations

import time

import numpy as np
import PIL.Image
import torch


def _load_image_encoder(path: str):
    from nanosam.utils.predictor import load_image_encoder_engine

    return load_image_encoder_engine(path)


def _load_mask_decoder(path: str):
    from nanosam.utils.predictor import load_mask_decoder_engine

    return load_mask_decoder_engine(path)


class NanoSamEncoder:
    """TensorRT image encoder only. Returns features [1,C,H,W] on CUDA."""

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
        if self.log:
            self.log.info(
                f"NanoSAM encoder engine={image_encoder} "
                f"size={self.image_encoder_size} feature={self.feature_size}"
            )
        self.engine = _load_image_encoder(image_encoder)
        self.last_timing = {"encoder_ms": 0.0}
        self.warmup()

    def warmup(self):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.encode(dummy)
        if self.log:
            self.log.info("NanoSAM encoder warmup complete")

    def preprocess(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """CPU resize/normalize; returns float32 CPU tensor [1,3,S,S]."""
        size = self.image_encoder_size
        image = PIL.Image.fromarray(rgb_uint8)
        image_mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
        image_std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]
        aspect_ratio = image.width / image.height
        if aspect_ratio >= 1:
            resize_width = size
            resize_height = int(size / aspect_ratio)
        else:
            resize_height = size
            resize_width = int(size * aspect_ratio)
        image_np = np.asarray(image.resize((resize_width, resize_height)))
        image_torch = torch.from_numpy(image_np).permute(2, 0, 1)
        image_torch = (image_torch.float() - image_mean) / image_std
        out = torch.zeros((1, 3, size, size), dtype=torch.float32)
        out[0, :, :resize_height, :resize_width] = image_torch
        return out

    @torch.inference_mode()
    def encode_tensor(
        self, image_tensor: torch.Tensor, *, synchronize: bool = True
    ) -> torch.Tensor:
        """Run TRT image encoder. Caller may wrap in ``torch.cuda.stream``."""
        if synchronize:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.to(
                device=self.dev, dtype=torch.float32, non_blocking=True
            )
        features = self.engine(image_tensor)
        if synchronize:
            torch.cuda.synchronize()
            self.last_timing = {"encoder_ms": (time.perf_counter() - t0) * 1000.0}
        return features

    @torch.inference_mode()
    def encode(self, rgb_uint8: np.ndarray, *, synchronize: bool = True) -> torch.Tensor:
        return self.encode_tensor(self.preprocess(rgb_uint8), synchronize=synchronize)


class NanoSamDecoder:
    """TensorRT mask decoder only. Prefers one batched enqueue for all boxes."""

    def __init__(
        self,
        mask_decoder: str,
        orig_image_encoder_size: int = 1024,
        device: str = "cuda",
        logger=None,
    ):
        self.dev = device
        self.orig_image_encoder_size = int(orig_image_encoder_size)
        self.feature_size = self.orig_image_encoder_size // 16
        self.mask_input_size = 4 * self.feature_size
        self.log = logger
        self._batch_ok = False
        if self.log:
            self.log.info(
                f"NanoSAM decoder engine={mask_decoder} "
                f"canvas={self.orig_image_encoder_size} "
                f"feature={self.feature_size} mask_in={self.mask_input_size}"
            )
        self.engine = _load_mask_decoder(mask_decoder)
        self.last_timing = {
            "decoder_ms": 0.0,
            "n_boxes": 0,
            "decoder_ms_per_box": 0.0,
            "decoder_mode": "serial",
        }
        self.warmup()

    def _warn(self, msg):
        if self.log:
            self.log.warn(msg)

    def _info(self, msg):
        if self.log:
            self.log.info(msg)

    def warmup(self):
        fs = self.feature_size
        feat = torch.zeros(1, 256, fs, fs, device=self.dev, dtype=torch.float32)
        boxes = [
            np.array([80, 80, 200, 200], dtype=np.float32),
            np.array([220, 120, 400, 360], dtype=np.float32),
        ]
        try:
            try:
                _ = self._predict_boxes_batched(feat, boxes, (480, 640))
                self._batch_ok = True
                self._info("NanoSAM mask decoder: batched prompts OK")
            except Exception as e:  # noqa: BLE001
                self._batch_ok = False
                self._warn(
                    f"NanoSAM mask decoder batch unsupported ({e}); "
                    "using serial decode. Rebuild with "
                    "perception_models/scripts/export_nanosam_mask_decoder_batched_onnx.py + "
                    "perception_models/scripts/build_nanosam_mask_decoder_batched_engine.sh"
                )
                _ = self._predict_box(feat, boxes[0], (480, 640))
            self._info(
                f"NanoSAM decoder warmup complete (mode="
                f"{'batched' if self._batch_ok else 'serial'})."
            )
        except Exception as e:  # noqa: BLE001
            self._warn(f"NanoSAM decoder warmup failed: {e}")
            raise

    def _scale_points(self, points_xy: np.ndarray, image_hw) -> np.ndarray:
        from nanosam.utils.predictor import preprocess_points

        return preprocess_points(
            points_xy, image_hw, self.orig_image_encoder_size
        )

    def _predict_box(
        self, features: torch.Tensor, box_xyxy: np.ndarray, image_hw
    ) -> torch.Tensor:
        from nanosam.utils.predictor import upscale_mask

        x0, y0, x1, y1 = [float(v) for v in box_xyxy]
        point_coords = np.array([[x0, y0], [x1, y1]], dtype=np.float32)
        point_labels = np.array([2, 3], dtype=np.float32)
        point_coords = self._scale_points(point_coords, image_hw)
        ms = self.mask_input_size
        pts = torch.from_numpy(point_coords).unsqueeze(0).to(
            device=self.dev, dtype=torch.float32
        )
        lbl = torch.from_numpy(point_labels).unsqueeze(0).to(
            device=self.dev, dtype=torch.float32
        )
        mask_input = torch.zeros(1, 1, ms, ms, device=self.dev, dtype=torch.float32)
        has_mask_input = torch.zeros(1, device=self.dev, dtype=torch.float32)
        _iou, low_res = self.engine(
            features, pts, lbl, mask_input, has_mask_input
        )
        hi = upscale_mask(low_res, image_hw, size=self.mask_input_size)
        return (hi[0, 0] > 0).float()

    def _predict_boxes_batched(
        self, features: torch.Tensor, boxes_xyxy, image_hw
    ) -> torch.Tensor:
        from nanosam.utils.predictor import upscale_mask

        k = len(boxes_xyxy)
        pts = np.zeros((k, 2, 2), dtype=np.float32)
        for i, box in enumerate(boxes_xyxy):
            x0, y0, x1, y1 = [float(v) for v in box]
            pts[i, 0] = (x0, y0)
            pts[i, 1] = (x1, y1)
        pts = self._scale_points(pts, image_hw)
        labels = np.tile(np.array([2.0, 3.0], dtype=np.float32), (k, 1))

        features_b = features.expand(k, -1, -1, -1).contiguous()
        point_coords = torch.from_numpy(pts).to(device=self.dev, dtype=torch.float32)
        point_labels = torch.from_numpy(labels).to(device=self.dev, dtype=torch.float32)
        ms = self.mask_input_size
        mask_input = torch.zeros(k, 1, ms, ms, device=self.dev, dtype=torch.float32)
        has_mask_input = torch.zeros(k, device=self.dev, dtype=torch.float32)

        _iou, low_res = self.engine(
            features_b,
            point_coords,
            point_labels,
            mask_input,
            has_mask_input,
        )
        if low_res.shape[1] > 1:
            low_res = low_res[:, :1]
        hi = upscale_mask(low_res, image_hw, size=self.mask_input_size)
        return (hi[:, 0] > 0).float()

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
        h, w = int(image_hw[0]), int(image_hw[1])
        refined = torch.zeros((n_classes, h, w), device=self.dev, dtype=torch.float32)
        self.last_timing = {
            "decoder_ms": 0.0,
            "n_boxes": 0,
            "decoder_ms_per_box": 0.0,
            "decoder_mode": "batched" if self._batch_ok else "serial",
        }
        if not boxes_xyxy:
            return refined

        valid_cls = []
        valid_boxes = []
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
            valid_cls.append(int(cls))
            valid_boxes.append(np.array([x0, y0, x1, y1], dtype=np.float32))
        if not valid_boxes:
            return refined

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mode = "serial"
        if self._batch_ok:
            try:
                masks = self._predict_boxes_batched(features, valid_boxes, (h, w))
                for i, cls in enumerate(valid_cls):
                    refined[cls] = torch.maximum(refined[cls], masks[i])
                mode = "batched"
            except Exception as e:  # noqa: BLE001
                self._warn(f"batched decode failed ({e}); serial fallback this frame")
                self._batch_ok = False
                mode = "serial"

        if mode == "serial":
            for cls, box in zip(valid_cls, valid_boxes):
                mask_t = self._predict_box(features, box, (h, w))
                refined[cls] = torch.maximum(refined[cls], mask_t)

        torch.cuda.synchronize()
        decoder_ms = (time.perf_counter() - t0) * 1000.0
        n_pred = len(valid_boxes)
        self.last_timing = {
            "decoder_ms": decoder_ms,
            "n_boxes": n_pred,
            "decoder_ms_per_box": (decoder_ms / n_pred) if n_pred else 0.0,
            "decoder_mode": mode,
        }
        return refined
