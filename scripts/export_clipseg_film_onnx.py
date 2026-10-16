#!/usr/bin/env python3
"""Export CLIPSeg FiLM decoder to fixed-shape ONNX.

  inputs:
    stacked_hs  [batch, L, S, D]
    cond        [num_prompts, cond_dim]
  output:
    probs       [batch, num_prompts, res, res]  (sigmoid logits)

Example:
  python3 scripts/export_clipseg_film_onnx.py --batch 3 --num-prompts 9 \\
    --output .../clipseg_film_rd16_224_b3_p9.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


class FilmDecode(nn.Module):
    """Batched image×prompt FiLM decode matching ClipSegParts._Dec (N>1 path)."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, stacked_hs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # stacked_hs: [N,L,S,D], cond: [P,C] → probs [N,P,H,W]
        n, l, s, d = stacked_hs.shape
        p = int(cond.shape[0])
        hs = tuple(
            stacked_hs[:, i]
            .unsqueeze(1)
            .expand(-1, p, -1, -1)
            .reshape(n * p, s, d)
            .contiguous()
            for i in range(l)
        )
        cond_b = cond.unsqueeze(0).expand(n, -1, -1).reshape(n * p, -1).contiguous()
        out = self.decoder(hs, cond_b).logits.sigmoid()
        return out.view(n, p, out.shape[-2], out.shape[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="CIDAS/clipseg-rd16")
    ap.add_argument("--input-res", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=9)
    ap.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Optional prompt strings (len must match --num-prompts if set)",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    n = max(1, int(args.batch))
    p = max(1, int(args.num_prompts))
    if args.prompts is not None:
        if len(args.prompts) != p:
            raise SystemExit(
                f"--prompts len {len(args.prompts)} != --num-prompts {p}"
            )
        prompt_list = list(args.prompts)
    else:
        prompt_list = [f"prompt_{i}" for i in range(p)]

    if args.output is None:
        stem = f"clipseg_film_rd16_{args.input_res}_b{n}_p{p}"
        args.output = Path(__file__).resolve().parents[1] / "data" / f"{stem}.onnx"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model} ...")
    model = CLIPSegForImageSegmentation.from_pretrained(args.model).eval()
    proc = CLIPSegProcessor.from_pretrained(args.model)
    tok = proc.tokenizer(prompt_list, return_tensors="pt", padding=True)
    # no_grad (not inference_mode): ONNX tracer needs autograd-compatible tensors.
    with torch.no_grad():
        cond = model.get_conditional_embeddings(
            batch_size=p,
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask,
        )
        pix = torch.zeros(n, 3, args.input_res, args.input_res)
        vo = model.clip.vision_model(
            pixel_values=pix, output_hidden_states=True, return_dict=True
        )
        extract = list(model.extract_layers)
        stacked = torch.stack(
            [vo.hidden_states[i + 1].contiguous() for i in extract], dim=1
        )

    wrap = FilmDecode(model.decoder).eval().cpu()
    stacked = stacked.detach().float().cpu().clone()
    cond = cond.detach().float().cpu().clone()
    with torch.no_grad():
        out = wrap(stacked, cond)
    print(
        f"extract={extract} batch={n} prompts={p} "
        f"hs={tuple(stacked.shape)} cond={tuple(cond.shape)} "
        f"probs={tuple(out.shape)}"
    )
    print(f"Exporting ONNX -> {args.output}")
    # Legacy TorchScript exporter (avoids onnxscript / dynamo dependency).
    from torch.onnx.utils import export as torchscript_onnx_export

    torchscript_onnx_export(
        wrap,
        (stacked, cond),
        str(args.output),
        input_names=["stacked_hs", "cond"],
        output_names=["probs"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print("done")


if __name__ == "__main__":
    main()
