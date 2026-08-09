"""Label / box / overlay helpers for CLIPSeg + NanoSAM (no ROS deps)."""

from __future__ import annotations

import cv2
import numpy as np

# Category colors: label i (1-indexed) -> PALETTE[(i-1) % len]; label 0 = none.
# Default order matches traversable -> green, non_traversable -> red, people -> blue.
PALETTE = [
    (0, 200, 0),
    (220, 0, 0),
    (0, 80, 255),
    (230, 230, 0),
    (0, 230, 230),
    (230, 0, 230),
]

# Default trav rgb8 colors (RGB). Mapper cost ≈ 1 - BT.601(luminance)/255.
DEFAULT_TRAVERSABILITY_COLORS = {
    "traversable": [0, 200, 0],
    "non_traversable": [220, 0, 0],
    "people": [0, 80, 255],
    "none": [255, 255, 255],
}


def parse_rgb_color(value, default: list[int]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float32)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
    return np.asarray(default, dtype=np.float32)


def traversability_colors_from_config(cfg: dict) -> dict[str, np.ndarray]:
    raw = cfg.get("traversability_colors") or {}
    return {
        key: parse_rgb_color(raw.get(key), DEFAULT_TRAVERSABILITY_COLORS[key])
        for key in DEFAULT_TRAVERSABILITY_COLORS
    }


def pack_traversability_rgb(
    group_maps,
    *,
    positive_groups,
    negative_groups,
    people_groups,
    group_has_prompts: list[bool],
    colors: dict[str, np.ndarray],
    threshold: float = 0.5,
):
    """Paint trav rgb8 from per-group maps [G,H,W] with role-based colors.

    Empty prompt groups are skipped (union is zero). Unlabeled pixels keep
    ``none``. Priority: non_traversable > people > traversable > none.
    Returns (rgb uint8 HxWx3, confidence HxW float tensor on group_maps device).
    """
    import torch

    def _union(idxs):
        sel = [
            i
            for i in idxs
            if 0 <= i < group_maps.shape[0] and group_has_prompts[i]
        ]
        if not sel:
            return group_maps.new_zeros(group_maps.shape[1:])
        return group_maps[sel].amax(dim=0)

    trav = _union(positive_groups)
    people = _union(people_groups)
    nontrav = _union(negative_groups)

    h, w = nontrav.shape
    dev = group_maps.device

    def _c(key: str) -> "torch.Tensor":
        return torch.as_tensor(colors[key], device=dev, dtype=torch.float32)

    rgb = _c("none").view(1, 1, 3).expand(h, w, 3).clone()
    tc = _c("traversable")
    pc = _c("people")
    nc = _c("non_traversable")

    if any(0 <= i < len(group_has_prompts) and group_has_prompts[i] for i in positive_groups):
        rgb[trav >= threshold] = tc

    if any(0 <= i < len(group_has_prompts) and group_has_prompts[i] for i in people_groups):
        rgb[people >= threshold] = pc

    if any(0 <= i < len(group_has_prompts) and group_has_prompts[i] for i in negative_groups):
        rgb[nontrav >= threshold] = nc

    conf = torch.maximum(torch.maximum(trav, people), nontrav)
    rgb_np = rgb.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
    return rgb_np, conf


def extract_boxes(binary_np: np.ndarray, min_area: float, max_boxes: int):
    """Per-class binary masks [n,H,W] (uint8) -> largest blobs as
    (class_idx, area, (x0, y0, x1, y1)), biggest first, capped at max_boxes."""
    cands = []
    for i in range(binary_np.shape[0]):
        ncomp, _, stats, _ = cv2.connectedComponentsWithStats(
            binary_np[i], connectivity=8
        )
        for c in range(1, ncomp):  # skip background label 0
            area = int(stats[c, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[c, cv2.CC_STAT_LEFT])
            y = int(stats[c, cv2.CC_STAT_TOP])
            bw = int(stats[c, cv2.CC_STAT_WIDTH])
            bh = int(stats[c, cv2.CC_STAT_HEIGHT])
            cands.append((i, area, (x, y, x + bw, y + bh)))
    cands.sort(key=lambda t: t[1], reverse=True)
    return cands[:max_boxes]


def labels_to_boxes(
    labels: np.ndarray, n_groups: int, min_area_frac: float, max_boxes: int
):
    """Label map [H,W] (0=none, i=group i-1) -> box prompts for SAM refinement.

    Only runs CC on labels that appear (skips empty groups).
    """
    h, w = labels.shape
    min_area = float(min_area_frac) * h * w
    cands = []
    for lab in np.unique(labels):
        lab_i = int(lab)
        if lab_i <= 0 or lab_i > n_groups:
            continue
        gi = lab_i - 1
        binary = (labels == lab_i).astype(np.uint8)
        ncomp, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for c in range(1, ncomp):
            area = int(stats[c, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[c, cv2.CC_STAT_LEFT])
            y = int(stats[c, cv2.CC_STAT_TOP])
            bw = int(stats[c, cv2.CC_STAT_WIDTH])
            bh = int(stats[c, cv2.CC_STAT_HEIGHT])
            cands.append((gi, area, (x, y, x + bw, y + bh)))
    cands.sort(key=lambda t: t[1], reverse=True)
    return cands[:max_boxes]


def labels_to_boxes_batch(
    labels_np_all: np.ndarray,
    n_groups: int,
    min_area_frac: float,
    max_boxes: int,
):
    """Batched ``labels_to_boxes`` over ``[N,H,W]`` (thread pool when N>1)."""
    n = int(labels_np_all.shape[0])
    if n <= 1:
        return [
            labels_to_boxes(labels_np_all[i], n_groups, min_area_frac, max_boxes)
            for i in range(n)
        ]
    from concurrent.futures import ThreadPoolExecutor

    def _one(i: int):
        return labels_to_boxes(
            labels_np_all[i], n_groups, min_area_frac, max_boxes
        )

    with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
        return list(pool.map(_one, range(n)))


def blend_overlay(
    base_rgb: np.ndarray, color_rgb: np.ndarray, conf: np.ndarray, alpha: float
) -> np.ndarray:
    """Blend color_rgb over base_rgb, weighted by conf*alpha. HxW(x3) -> uint8."""
    strength = (conf * alpha)[:, :, None]
    out = (
        base_rgb.astype(np.float32) * (1.0 - strength)
        + color_rgb.astype(np.float32) * strength
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def probs_to_labels(probs, threshold: float):
    """Per-group probs [n,H,W] -> (labels uint8 [H,W], confidence float [H,W])."""
    import torch  # local: keep module importable without torch at load time

    maxv, arg = probs.max(dim=0)
    labels = (arg + 1).to(torch.uint8)
    labels = torch.where(maxv >= threshold, labels, torch.zeros_like(labels))
    return labels.cpu().numpy(), maxv.float().cpu().numpy()


def labels_to_color(labels: np.ndarray, n_groups: int) -> np.ndarray:
    """Color a label map [H,W] via PALETTE (label 0 stays black). -> HxWx3 uint8."""
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for i in range(n_groups):
        out[labels == (i + 1)] = PALETTE[i % len(PALETTE)]
    return out
