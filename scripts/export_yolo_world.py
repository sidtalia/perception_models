#!/usr/bin/env python3
"""Export YOLO-World with stashed classes (prompt-then-detect).

1. Load yolov8*-world.pt
2. set_classes(classes)  — stash CLIP text embeddings
3. Save specialized ``*_stashed.pt`` (production torch FP16 artifact)
4. Optionally try ultralytics export(format=onnx|engine)

ONNX/TRT note
-------------
Ultralytics WorldModel export currently fails on ``adaptive_max_pool2d``
(unsupported in ONNX). Default ``--format pt`` is the supported path.
Point SSoT ``yolo_world.model`` at the stashed ``.pt``.

Usage (inside mushr_jazzy)::

  python3 /root/colcon_ws/src/perception_models/scripts/export_yolo_world.py \\
    --model yolov8s-world.pt \\
    --classes person chair door \\
    --out-dir /root/colcon_ws/src/perception_models/data \\
    --format pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolov8s-world.pt")
    p.add_argument("--classes", nargs="+", default=["person"])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    p.add_argument("--format", choices=("pt", "onnx", "engine"), default="pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--no-half", action="store_false", dest="half")
    args = p.parse_args()

    from ultralytics import YOLO

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.model} ...")
    model = YOLO(args.model)
    classes = list(args.classes)
    if "person" not in classes:
        classes = ["person"] + classes
    print(f"set_classes / stash embeddings: {classes}")
    model.set_classes(classes)

    stem = Path(args.model).stem + "_stashed"
    pt_path = args.out_dir / f"{stem}.pt"
    model.save(str(pt_path))
    print(f"saved specialized weights: {pt_path}")

    if args.format == "pt":
        print(
            "TRT skipped (--format pt). ONNX/engine often fail on "
            "WorldModel adaptive_max_pool2d; use torch FP16 + this .pt."
        )
        return

    print(f"export format={args.format} imgsz={args.imgsz} half={args.half} ...")
    try:
        out = model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            device=0,
        )
        print(f"export ok: {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"export FAILED ({exc})")
        print(
            "Keep the stashed .pt for torch FP16 runtime. "
            "TRT for YOLO-World needs a custom ONNX path (no adaptive_max_pool2d)."
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
