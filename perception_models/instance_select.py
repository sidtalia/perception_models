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
