#!/usr/bin/env python3
"""Bench YOLO-World detect FPS with cached (stashed) prompts.

Sweeps model × imgsz by default. Classes are stashed once per config
(``set_classes``); measured path is detect-only.

Usage (mushr_jazzy)::

  python3 /root/colcon_ws/src/perception_models/scripts/bench_yolo_world.py
  python3 .../bench_yolo_world.py --models yolov8n-world.pt yolov8s-world.pt \\
    --imgsz 320 416 640 --iters 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def _bench_one(
    *,
    model: str,
    classes: list[str],
    imgsz: int,
    warmup: int,
    iters: int,
    conf: float,
    half: bool,
    h: int,
    w: int,
) -> dict:
    from perception_models.yolo_world.shared import YoloWorldParts, YoloWorldPromptLayout

    parts = YoloWorldParts(
        model_name=model,
        prompts=YoloWorldPromptLayout(classes=classes),
        conf=conf,
        imgsz=imgsz,
        half=half,
        logger=lambda _m: None,
    )
    parts.warmup()
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    for _ in range(warmup):
        parts.detect_boxes(rgb)
    torch.cuda.synchronize()

    times = []
    n_boxes = 0
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        boxes = parts.detect_boxes(rgb)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        n_boxes = len(boxes)
    arr = np.asarray(times, dtype=np.float64)
    mean_ms = float(arr.mean())
    return {
        "model": model,
        "imgsz": imgsz,
        "mean_ms": mean_ms,
        "p50_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "last_boxes": n_boxes,
        "input_hw": (h, w),
        "half": half,
        "n": iters,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n-world.pt", "yolov8s-world.pt"],
    )
    p.add_argument("--imgsz", nargs="+", type=int, default=[320, 416, 640])
    p.add_argument("--classes", nargs="+", default=["person", "chair", "door"])
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--no-half", action="store_true")
    p.add_argument(
        "--input-hw",
        type=int,
        nargs=2,
        default=[360, 640],
        metavar=("H", "W"),
        help="Synthetic RGB size (CrowdBot/RealSense-like default 360x640)",
    )
    args = p.parse_args()
    half = not args.no_half
    h, w = int(args.input_hw[0]), int(args.input_hw[1])

    print(
        f"YOLO-World FPS (cached prompts={args.classes}, half={half}, "
        f"input={h}x{w}, warmup={args.warmup}, iters={args.iters})"
    )
    print(f"{'model':22} {'imgsz':>5} {'ms':>7} {'p50':>7} {'p90':>7} {'FPS':>7} boxes")
    rows = []
    for model in args.models:
        for imgsz in args.imgsz:
            try:
                r = _bench_one(
                    model=model,
                    classes=list(args.classes),
                    imgsz=int(imgsz),
                    warmup=args.warmup,
                    iters=args.iters,
                    conf=args.conf,
                    half=half,
                    h=h,
                    w=w,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{model:22} {imgsz:5d} FAIL ({exc})")
                continue
            rows.append(r)
            print(
                f"{r['model']:22} {r['imgsz']:5d} {r['mean_ms']:7.1f} "
                f"{r['p50_ms']:7.1f} {r['p90_ms']:7.1f} {r['fps']:7.1f} "
                f"{r['last_boxes']}"
            )

    if rows:
        best = max(rows, key=lambda x: x["fps"])
        print(
            f"best FPS: {best['fps']:.1f}  "
            f"({best['model']} imgsz={best['imgsz']} → {best['mean_ms']:.1f} ms)"
        )


if __name__ == "__main__":
    main()
