#!/usr/bin/env python3
"""Benchmark CLIPSeg vision encode (+ optional TRT) and FiLM decode on Orin.

Examples (inside mushr_jazzy):

  # Torch baseline (compiled + eager)
  python3 /root/colcon_ws/src/perception_models/scripts/bench_clipseg.py

  # A/B vs a TensorRT vision engine
  python3 /root/colcon_ws/src/perception_models/scripts/bench_clipseg.py \\
    --trt-engine /root/colcon_ws/src/perception_models/data/clipseg_vision_rd16_224.engine
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from perception_models.clipseg.shared import ClipSegParts, PromptLayout  # noqa: E402


def _time_cuda(fn, n: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


def _load_trt(path: str, input_name: str = "pixel_values", output_name: str = "hidden_states"):
    import tensorrt as trt
    from torch2trt import TRTModule

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    return TRTModule(engine=engine, input_names=[input_name], output_names=[output_name])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="CIDAS/clipseg-rd16")
    ap.add_argument("--input-res", type=int, default=224)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--compile-mode", default="reduce-overhead")
    ap.add_argument(
        "--trt-engine",
        default="",
        help="Optional TensorRT vision engine path for A/B",
    )
    ap.add_argument(
        "--go-threshold",
        type=float,
        default=0.25,
        help="Fraction faster than compiled torch for GO (default 0.25 = 25%%)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    layout = PromptLayout(
        subprompts=[
            "floor",
            "carpet",
            "concrete",
            "furniture",
            "wall",
            "pillar",
            "person",
            "car",
        ],
        group_names=["traversable", "non_traversable", "people"],
        group_ids=[0, 0, 0, 1, 1, 1, 2, 2],
        suppress_prompts=["sky"],
    )

    def log(msg: str):
        print(msg, flush=True)

    parts = ClipSegParts(
        args.model,
        layout,
        input_res=args.input_res,
        compile_mode="none",
        device="cuda",
        logger=log,
    )
    enc_eager = parts.build_encoder()
    enc_eager = parts.warmup_encoder(enc_eager)
    # Rebuild compiled encoder separately for A/B.
    parts_c = ClipSegParts(
        args.model,
        layout,
        input_res=args.input_res,
        compile_mode=args.compile_mode,
        device="cuda",
        logger=log,
    )
    # Share weights already on GPU via same HF load — rebuild is fine for spike.
    enc_compiled = parts_c.build_encoder()
    enc_compiled = parts_c.warmup_encoder(enc_compiled)
    dec = parts.warmup_decoder(parts.build_decoder())

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    h, w = rgb.shape[:2]

    pix = parts.preprocess_rgb(rgb)  # [1,3,res,res]

    def encode_eager():
        with torch.autocast("cuda", dtype=torch.float16):
            return enc_eager(pix)

    def encode_compiled():
        with torch.autocast("cuda", dtype=torch.float16):
            return enc_compiled(pix)

    hs_ref = encode_compiled().float()

    def film():
        return parts.decode_probs(
            hs_ref, dec, height=h, width=w, aggregate="max"
        )

    pre_ms = _time_cuda(lambda: parts.preprocess_rgb(rgb), args.iters, args.warmup)
    eager_ms = _time_cuda(encode_eager, args.iters, args.warmup)
    compiled_ms = _time_cuda(encode_compiled, args.iters, args.warmup)
    film_ms = _time_cuda(film, args.iters, args.warmup)

    print("\n=== Torch baseline ===")
    print(f"preprocess     {pre_ms:6.2f} ms")
    print(f"encode eager   {eager_ms:6.2f} ms")
    print(f"encode compile {compiled_ms:6.2f} ms  (mode={args.compile_mode})")
    print(f"film_decode    {film_ms:6.2f} ms")
    print(f"hs shape {tuple(hs_ref.shape)} dtype={hs_ref.dtype}")

    if not args.trt_engine:
        print("\n(no --trt-engine; skip A/B)")
        return

    trt = _load_trt(args.trt_engine)
    # TRTModule usually wants float32/fp16 matching build; feed same pix.
    pix_trt = pix.float().contiguous()

    def encode_trt():
        out = trt(pix_trt)
        return out[0] if isinstance(out, (tuple, list)) else out

    # Warmup TRT
    for _ in range(args.warmup):
        encode_trt()
    torch.cuda.synchronize()
    trt_ms = _time_cuda(encode_trt, args.iters, 0)

    hs_trt = encode_trt().float()
    if hs_trt.shape != hs_ref.shape:
        # Some exports keep batch dim.
        if hs_trt.dim() == 4 and hs_trt.shape[0] == 1:
            hs_trt = hs_trt[0]
        if hs_trt.shape != hs_ref.shape and hs_trt.dim() == 4:
            # [1,L,S,D] vs [L,S,D]
            hs_trt = hs_trt.squeeze(0)

    # Align shapes for error
    if hs_trt.shape != hs_ref.shape:
        print(f"WARNING: TRT hs shape {tuple(hs_trt.shape)} != torch {tuple(hs_ref.shape)}")
        # Try permute if [L,1,S,D]
        if hs_trt.numel() == hs_ref.numel():
            hs_trt = hs_trt.reshape(hs_ref.shape)

    diff = (hs_trt.cuda() - hs_ref).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    # Cosine over flattened
    a = hs_ref.reshape(-1).float()
    b = hs_trt.cuda().reshape(-1).float()
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

    with torch.inference_mode():
        probs_t = parts.decode_probs(
            hs_ref, dec, height=h, width=w, aggregate="max"
        )
        probs_r = parts.decode_probs(
            hs_trt.cuda(), dec, height=h, width=w, aggregate="max"
        )
    probs_max = float((probs_t - probs_r).abs().max())

    speedup = compiled_ms / trt_ms if trt_ms > 0 else 0.0
    frac_faster = 1.0 - (trt_ms / compiled_ms) if compiled_ms > 0 else 0.0

    print("\n=== TRT A/B ===")
    print(f"encode TRT     {trt_ms:6.2f} ms")
    print(f"vs compiled    {speedup:6.2f}x  ({100.0 * frac_faster:+.1f}% wall)")
    print(f"hs max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} cosine={cos:.6f}")
    print(f"FiLM probs max_abs={probs_max:.5f}")

    go = frac_faster >= args.go_threshold and cos > 0.99 and probs_max < 0.15
    print("\n=== Verdict ===")
    if go:
        print(
            f"GO — TRT encode >= {100 * args.go_threshold:.0f}% faster than "
            f"compiled torch and features look sane."
        )
    else:
        reasons = []
        if frac_faster < args.go_threshold:
            reasons.append(
                f"speed {100 * frac_faster:.1f}% < {100 * args.go_threshold:.0f}% threshold"
            )
        if cos <= 0.99:
            reasons.append(f"cosine {cos:.6f} <= 0.99")
        if probs_max >= 0.15:
            reasons.append(f"probs max_abs {probs_max:.5f} >= 0.15")
        print("NO-GO — " + "; ".join(reasons) if reasons else "NO-GO")
    # Machine-readable for scripts
    print(f"VERDICT={'GO' if go else 'NO_GO'}")


if __name__ == "__main__":
    main()
