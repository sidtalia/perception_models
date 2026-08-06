#!/usr/bin/env python3
"""Export CLIPSeg vision tower (+ extract_layers stack) to fixed-shape ONNX.

  input  pixel_values  [batch,3,res,res]
  output hidden_states [batch,L,S,D]  (or [L,S,D] when batch=1)

Example:
  python3 scripts/export_clipseg_vision_onnx.py --batch 3 \\
    --output .../clipseg_vision_rd16_224_b3.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPSegForImageSegmentation


class VisionExtract(nn.Module):
    def __init__(self, vision_model: nn.Module, extract_layers: list[int], batch: int):
        super().__init__()
        self.vision = vision_model
        self.extract = list(extract_layers)
        self.batch = int(batch)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vo = self.vision(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        layers = [vo.hidden_states[i + 1].contiguous() for i in self.extract]
        stacked = torch.stack(layers, dim=1)  # [N,L,S,D]
        if self.batch == 1:
            return stacked.squeeze(0)
        return stacked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="CIDAS/clipseg-rd16")
    ap.add_argument("--input-res", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if args.output is None:
        stem = f"clipseg_vision_rd16_{args.input_res}"
        if args.batch > 1:
            stem += f"_b{args.batch}"
        args.output = Path(__file__).resolve().parents[1] / "data" / f"{stem}.onnx"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model} ...")
    model = CLIPSegForImageSegmentation.from_pretrained(args.model).eval()
    extract = list(model.extract_layers)
    wrap = VisionExtract(model.clip.vision_model, extract, args.batch).eval().cpu()
    dummy = torch.zeros(args.batch, 3, args.input_res, args.input_res, dtype=torch.float32)
    with torch.inference_mode():
        out = wrap(dummy)
    print(f"extract_layers={extract} batch={args.batch} output={tuple(out.shape)}")
    print(f"Exporting ONNX -> {args.output}")
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
    print("done")


if __name__ == "__main__":
    main()
