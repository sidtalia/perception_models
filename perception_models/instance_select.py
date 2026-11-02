"""Prompted-object instance proposals from CLIPSeg probability maps.

Each prompt is thresholded *independently*, so overlapping prompts like "mug"
and "red mug" both work, unlike the exclusive argmax label map the
traversability path used. Every connected component becomes a candidate
instance. Its score is the mean CLIPSeg probability inside the component. Its
box (padded, in full-image pixels) is the NanoSAM box prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Instance:
    prompt_idx: int
    score: float
    area_frac: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in full-image pixels


def propose_instances(
    probs: np.ndarray,
    image_hw: tuple[int, int],
    *,
    threshold: float = 0.4,
    min_area_frac: float = 0.0005,
    max_instances_per_prompt: int = 2,
    box_pad_frac: float = 0.08,
) -> list[Instance]:
    """Instances for one image, best first.

    Args:
        probs: ``[P, h, w]`` CLIPSeg probabilities at native FiLM resolution
            (the image was resized, not letterboxed, to ``h x w``).
        image_hw: full image ``(H, W)`` the boxes are scaled to.
        threshold: per-prompt probability threshold.
        min_area_frac: drop components smaller than this fraction of the map.
        max_instances_per_prompt: keep at most this many components per prompt.
        box_pad_frac: pad each box by this fraction of its size on every side
            (CLIPSeg maps are coarse and tend to under-cover object edges).
    """
    if probs.ndim != 3:
        raise ValueError(f"probs must be [P,h,w], got {probs.shape}")
    n_p, h, w = probs.shape
    big_h, big_w = int(image_hw[0]), int(image_hw[1])
    sx, sy = big_w / float(w), big_h / float(h)
    min_area = max(1.0, min_area_frac * h * w)
    out: list[Instance] = []
    for p in range(n_p):
        pm = probs[p]
        binary = (pm >= threshold).astype(np.uint8)
        if not binary.any():
            continue
        ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cands: list[Instance] = []
        for c in range(1, ncomp):
            area = int(stats[c, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            score = float(pm[labels == c].mean())
            x = stats[c, cv2.CC_STAT_LEFT]
            y = stats[c, cv2.CC_STAT_TOP]
            bw = stats[c, cv2.CC_STAT_WIDTH]
            bh = stats[c, cv2.CC_STAT_HEIGHT]
            px, py = box_pad_frac * bw, box_pad_frac * bh
            x0 = max(0.0, (x - px) * sx)
            y0 = max(0.0, (y - py) * sy)
            x1 = min(big_w - 1.0, (x + bw + px) * sx)
            y1 = min(big_h - 1.0, (y + bh + py) * sy)
            cands.append(Instance(p, score, area / float(h * w), (x0, y0, x1, y1)))
        cands.sort(key=lambda i: (i.score, i.area_frac), reverse=True)
        out.extend(cands[:max_instances_per_prompt])
    out.sort(key=lambda i: (i.score, i.area_frac), reverse=True)
    return out


def instances_from_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_idx: np.ndarray,
    image_hw: tuple[int, int],
    *,
    min_score: float = 0.0,
    max_instances_per_prompt: int = 2,
    box_pad_frac: float = 0.02,
) -> list[Instance]:
    """Instances from detector boxes (YOLO-World), best first.

    Args:
        boxes: ``[K, 4]`` xyxy boxes in full-image pixels.
        scores: ``[K]`` detector confidences.
        class_idx: ``[K]`` prompt index of each box.
        image_hw: full image ``(H, W)``; boxes are clipped to it.
        min_score: drop boxes below this confidence.
        max_instances_per_prompt: keep at most this many boxes per prompt.
        box_pad_frac: pad each box by this fraction of its size on every side.
            Detector boxes are tight, so the default is small; a little slack
            keeps SAM from clipping object edges.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    class_idx = np.asarray(class_idx, dtype=np.int64).reshape(-1)
    if not (len(boxes) == len(scores) == len(class_idx)):
        raise ValueError(f"boxes/scores/class_idx lengths differ: {len(boxes)}, {len(scores)}, {len(class_idx)}")
    big_h, big_w = float(image_hw[0]), float(image_hw[1])
    per_prompt: dict[int, list[Instance]] = {}
    for (x0, y0, x1, y1), s, p in zip(boxes, scores, class_idx):
        if s < min_score or not np.isfinite([x0, y0, x1, y1]).all():
            continue
        px, py = box_pad_frac * (x1 - x0), box_pad_frac * (y1 - y0)
        b = (max(0.0, x0 - px), max(0.0, y0 - py), min(big_w - 1.0, x1 + px), min(big_h - 1.0, y1 + py))
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        area = (b[2] - b[0]) * (b[3] - b[1]) / (big_h * big_w)
        per_prompt.setdefault(int(p), []).append(Instance(int(p), float(s), float(area), b))
    out: list[Instance] = []
    for cands in per_prompt.values():
        cands.sort(key=lambda i: i.score, reverse=True)
        out.extend(cands[:max_instances_per_prompt])
    out.sort(key=lambda i: i.score, reverse=True)
    return out
