#!/usr/bin/env python3
"""Export CLIPSeg vision tower (+ extract_layers stack) to fixed-shape ONNX.

Matches perception_models.clipseg.shared encoder output for batch=1:
  input  pixel_values  [1,3,res,res]
  output hidden_states [L,S,D]   (L = len(extract_layers))

Example:
  python3 scripts/export_clipseg_vision_onnx.py \\
    --output /root/colcon_ws/src/perception_models/data/clipseg_vision_rd16_224.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPSegForImageSegmentation


class VisionExtract(nn.Module):
    """Vision model → stacked selected hidden states [L,S,D]."""

    def __init__(self, vision_model: nn.Module, extract_layers: list[int]):
        super().__init__()
        self.vision = vision_model
        self.extract = list(extract_layers)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vo = self.vision(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states[0] = embeddings; extract indices are 0-based into blocks.
        layers = [
            vo.hidden_states[i + 1].squeeze(0).contiguous() for i in self.extract
        ]
        return torch.stack(layers, dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="CIDAS/clipseg-rd16")
    ap.add_argument("--input-res", type=int, default=224)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "clipseg_vision_rd16_224.onnx",
    )
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model = CLIPSegForImageSegmentation.from_pretrained(args.model).eval()
    extract = list(model.extract_layers)
    wrap = VisionExtract(model.clip.vision_model, extract).eval().cpu()

    dummy = torch.zeros(1, 3, args.input_res, args.input_res, dtype=torch.float32)
    with torch.inference_mode():
        out = wrap(dummy)
    print(f"extract_layers={extract}  output shape={tuple(out.shape)}")

    print(f"Exporting ONNX opset={args.opset} -> {args.output}")
    torch.onnx.export(
        wrap,
        dummy,
        str(args.output),
        input_names=["pixel_values"],
        output_names=["hidden_states"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
