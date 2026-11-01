#!/usr/bin/env python3
"""Export MobileSAM's TinyViT image encoder to ONNX (fixed batch N, 1024²).

Use this when the distilled NanoSAM ResNet18 encoder ONNX is unavailable; it is
only published on Google Drive. TinyViT is the teacher that ResNet18 was
distilled from. It produces the same ``[N,256,64,64]`` embeddings the MobileSAM
mask decoder expects, and it is more accurate at the cost of a few ms.

TinyViT's window reshapes bake the traced batch size into the graph, so the
engine is built for a fixed image batch (= number of cameras), like the CLIPSeg
vision engine. I/O names match ``nanosam.utils.predictor.load_image_encoder_engine``:
``image`` ``[N,3,1024,1024]`` (SAM-normalized, top-left letterboxed) →
``image_embeddings``.

    python3 scripts/export_mobilesam_image_encoder_onnx.py \\
        --checkpoint ../nanosam-src/assets/mobile_sam.pt --batch 1 --output data/mobilesam_tinyvit_1024_b1.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=1, help="image batch (number of cameras)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    nanosam_src = Path(__file__).resolve().parents[2] / "nanosam-src"
    if nanosam_src.is_dir():
        sys.path.insert(0, str(nanosam_src))
    from nanosam.mobile_sam import sam_model_registry

    sam = sam_model_registry["vit_t"](checkpoint=args.checkpoint).eval()
    enc = sam.image_encoder
    dummy = torch.zeros(args.batch, 3, 1024, 1024)
    from torch.onnx.utils import export as torchscript_onnx_export  # legacy exporter (no onnxscript)

    with torch.inference_mode():
        torchscript_onnx_export(
            enc,
            (dummy,),
            args.output,
            input_names=["image"],
            output_names=["image_embeddings"],
            opset_version=args.opset,
            do_constant_folding=True,
        )
    print("wrote", args.output)


if __name__ == "__main__":
    main()
