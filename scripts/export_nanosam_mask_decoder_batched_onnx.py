#!/usr/bin/env python3
"""Re-export MobileSAM mask decoder ONNX with a true dynamic *box batch* axis.

Stock NanoSAM / SamOnnxModel keeps prompt batch fixed at 1 (only num_points is
dynamic), so TRT builds a static engine and ``trtexec`` fails when testing
optShapes>1. This export uses a thin wrapper that:

  * expects ``image_embeddings`` already shaped ``[B,C,H,W]`` (caller expands
    the single encoder feature map),
  * skips ``repeat_interleave`` that would turn B into B*B,
  * marks dim-0 of all I/O as dynamic.

Usage:
  python3 scripts/export_nanosam_mask_decoder_batched_onnx.py \\
    --checkpoint /root/colcon_ws/src/nanosam-src/assets/mobile_sam.pt \\
    --output /root/colcon_ws/src/perception_models/data/mobile_sam_mask_decoder_batched.onnx
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/assets/mobile_sam.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-type", default="vit_t")
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Encoder canvas size (1024 -> embed 64; 512 -> embed 32). "
        "Must match the image encoder engine.",
    )
    parser.add_argument(
        "--trace-batch",
        type=int,
        default=2,
        help="Batch size used while tracing (must be >1 so dim0 stays dynamic)",
    )
    args = parser.parse_args()
    if args.trace_batch < 2:
        raise SystemExit("--trace-batch must be >= 2")
    if args.image_size % 16 != 0:
        raise SystemExit("--image-size must be divisible by 16")

    src_root = Path(__file__).resolve().parents[1]
    nanosam_src = src_root.parent / "nanosam-src"
    if nanosam_src.is_dir():
        sys.path.insert(0, str(nanosam_src))

    from nanosam.mobile_sam import sam_model_registry

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        for cand in (
            src_root.parent / "nanosam-src" / "assets" / "mobile_sam.pt",
            Path("/root/colcon_ws/src/nanosam-src/assets/mobile_sam.pt"),
            Path("/root/colcon_ws/mobile_sam.pt"),
        ):
            if cand.is_file():
                ckpt = cand
                break
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    print(f"Loading MobileSAM ({args.model_type}) from {ckpt} ...")
    sam = sam_model_registry[args.model_type](checkpoint=str(ckpt))
    # Retarget prompt encoder spatial size to match a NanoSAM encoder at image_size.
    image_size = int(args.image_size)
    emb = image_size // 16
    sam.prompt_encoder.image_embedding_size = (emb, emb)
    sam.prompt_encoder.input_image_size = (image_size, image_size)
    sam.prompt_encoder.mask_input_size = (4 * emb, 4 * emb)
    # Point coords in the ONNX wrapper are normalized by img_size.
    sam.image_encoder.img_size = image_size
    print(f"Retargeted decoder spatials: image={image_size} embed={emb}x{emb}")

    onnx_model = BatchedMaskDecoderOnnx(sam).eval()

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    b = int(args.trace_batch)

    dummy_inputs = {
        "image_embeddings": torch.randn(b, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(
            low=0, high=image_size, size=(b, 2, 2), dtype=torch.float
        ),
        "point_labels": torch.randint(low=0, high=4, size=(b, 2), dtype=torch.float),
        "mask_input": torch.randn(b, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.ones(b, dtype=torch.float),
    }
    dynamic_axes = {
        "image_embeddings": {0: "batch_size"},
        "point_coords": {0: "batch_size", 1: "num_points"},
        "point_labels": {0: "batch_size", 1: "num_points"},
        "mask_input": {0: "batch_size"},
        "has_mask_input": {0: "batch_size"},
        "iou_predictions": {0: "batch_size"},
        "low_res_masks": {0: "batch_size"},
    }

    with torch.no_grad():
        _ = onnx_model(**dummy_inputs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(
            f"Exporting {out} (embed={embed_dim}x{embed_size}, "
            f"trace_batch={b}, image_embeddings batched) ..."
        )
        from torch.onnx.utils import export as torchscript_onnx_export

        torchscript_onnx_export(
            onnx_model,
            tuple(dummy_inputs.values()),
            str(out),
            export_params=True,
            verbose=False,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=["iou_predictions", "low_res_masks"],
            dynamic_axes=dynamic_axes,
        )
    print(f"Wrote {out}")
    print(
        "Next: bash scripts/build_nanosam_mask_decoder_batched_engine.sh "
        f"{out}  # [onnx] [engine] [opt_batch] [max_batch]"
    )


class BatchedMaskDecoderOnnx(nn.Module):
    """Prompt encoder + mask decoder; image_embeddings already [B,C,H,W]."""

    def __init__(self, sam):
        super().__init__()
        self.mask_decoder = sam.mask_decoder
        self.model = sam
        self.img_size = sam.image_encoder.img_size

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor):
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)
        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )
        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[
                i
            ].weight * (point_labels == i)
        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor):
        # has_mask_input: [B] -> [B,1,1,1]
        has = has_mask_input.view(-1, 1, 1, 1)
        mask_embedding = has * self.model.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (1 - has) * self.model.prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        )
        return mask_embedding

    def _select_best_mask(
        self, masks: torch.Tensor, iou_preds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # masks [B, M, H, W], iou [B, M] -> take best mask token per batch item
        best_idx = torch.argmax(iou_preds, dim=1)
        batch = torch.arange(masks.shape[0], device=masks.device)
        masks = masks[batch, best_idx].unsqueeze(1)
        iou_preds = iou_preds[batch, best_idx].unsqueeze(1)
        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ):
        sparse = self._embed_points(point_coords, point_labels)
        dense = self._embed_masks(mask_input, has_mask_input)

        # Inlined MaskDecoder.predict_masks without repeat_interleave:
        # image_embeddings is already [B,C,H,W].
        output_tokens = torch.cat(
            [self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight],
            dim=0,
        )
        output_tokens = output_tokens.unsqueeze(0).expand(sparse.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse), dim=1)

        src = image_embeddings + dense
        image_pe = self.model.prompt_encoder.get_dense_pe()
        pos_src = image_pe.expand(src.shape[0], -1, -1, -1)
        b, c, h, w = src.shape

        hs, src = self.mask_decoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.mask_decoder.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled = self.mask_decoder.output_upscaling(src)
        hyper_in_list = [
            self.mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.mask_decoder.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.mask_decoder.iou_prediction_head(iou_token_out)

        masks, iou_pred = self._select_best_mask(masks, iou_pred)
        return iou_pred, masks


if __name__ == "__main__":
    main()
