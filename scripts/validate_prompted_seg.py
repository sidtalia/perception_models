#!/usr/bin/env python3
"""Offline check of the prompted-segmentation hot path (no dora).

Runs exactly what ``dora_seg_refine`` does per frame: FiLM ∥ SAM-encode on two
CUDA streams, then instance proposals, then one batched SAM decode. It also
swaps prompts through the background ``PromptBank`` while the loop runs.

    python scripts/validate_prompted_seg.py --config seg.json --image frame.npz \\
        --prompts "a yellow mustard bottle" "a red mug" --out /tmp/overlay.png
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from perception_models.clipseg.shared import ClipSegParts, prompt_layout_from_config
from perception_models.clipseg.trt import (
    TrtFilmDecoder,
    film_engine_error,
    load_clipseg_film_engine,
    load_clipseg_vision_engine,
    trt_encode_hs,
)
from perception_models.dora_seg_refine import layout_for
from perception_models.instance_select import propose_instances
from perception_models.nanosam_trt import NanoSamDecoder, NanoSamEncoder
from perception_models.prompt_bank import PromptBank
from perception_models.seg_ops import PALETTE, blend_overlay


class _L:
    def info(self, m):
        print(m)

    warning = warn = info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True, help=".npz with 'rgb' (HxWx3 uint8) or an image file")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--swap-to", nargs="*", default=None, help="prompts to switch to mid-run")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    if args.image.endswith(".npz"):
        rgb = np.load(args.image)["rgb"]
    else:
        import cv2

        rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    parts = ClipSegParts(cfg.get("model_name", "CIDAS/clipseg-rd16"), prompt_layout_from_config(cfg),
                         input_res=int(cfg.get("input_res", 224)), compile_mode="none", device="cuda", logger=print)  # fmt: skip
    vis = load_clipseg_vision_engine(cfg["clipseg_vision_engine"])
    eager = parts.build_decoder()
    dec, slots = eager, 8
    if cfg.get("film_backend", "auto") != "torch":
        trt_dec = TrtFilmDecoder(load_clipseg_film_engine(cfg["clipseg_film_engine"]))
        try:
            err = film_engine_error(trt_dec, eager, [1, 3, (parts.res // 16) ** 2 + 1, 768],
                                    int(parts.cond.shape[1]), int(trt_dec._num_prompts))  # fmt: skip
        except Exception as e:  # noqa: BLE001 - engine built for another shape
            err = float("inf")
            print(f"FiLM TRT unusable: {e}")
        print(f"FiLM TRT self-check max err {err:.4f}")
        if err <= float(cfg.get("film_check_tol", 0.05)):
            dec, slots = trt_dec, trt_dec._num_prompts
    print("FiLM backend:", "trt" if dec is not eager else "eager fp16")
    size = int(cfg.get("sam_image_size", 1024))
    sam_enc = NanoSamEncoder(cfg["sam_image_encoder"], image_encoder_size=size, logger=_L())
    sam_dec = NanoSamDecoder(cfg["sam_mask_decoder"], orig_image_encoder_size=size, logger=_L(), max_batch_k=16)
    s_film, s_sam, s_text = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    bank = PromptBank(lambda ps: parts.embed_prompts(ps, stream=s_text), max_prompts=slots)
    bank.set_blocking(args.prompts)

    timings, versions = [], []
    insts, masks = [], None
    for it in range(args.iters):
        if args.swap_to and it == args.iters // 2:
            bank.request(args.swap_to)
        t0 = time.perf_counter()
        snap = bank.current()
        hs = trt_encode_hs(vis, parts.preprocess_rgb_batch([rgb]))
        s_film.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_film):
            probs = parts.decode_probs_batch(hs, dec, aggregate="max", cond=snap.cond, layout=layout_for(snap.prompts))
        sam_in = sam_enc.preprocess_batch([rgb])
        with torch.cuda.stream(s_sam):
            feats = sam_enc.encode_tensor(sam_in, synchronize=False)
        with torch.cuda.stream(s_film):
            probs_np = probs.float().cpu().numpy()
        insts = propose_instances(probs_np[0], (h, w), threshold=float(cfg.get("threshold", 0.4)))
        s_sam.synchronize()
        masks = sam_dec.decode_multi(feats, [(list(range(len(insts))), [i.box for i in insts], max(1, len(insts)),
                                               (h, w))])[0]  # fmt: skip
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1e3)
        versions.append(snap.version)
    bank.close()

    t = np.array(timings[5:])
    print(f"frame time: median {np.median(t):.1f} ms  p90 {np.percentile(t, 90):.1f} ms  (incl. CLIPSeg vision)")
    print(f"prompt versions seen: {sorted(set(versions))}  final prompts: {list(snap.prompts)}")
    for i in insts:
        area = int(masks[insts.index(i)].sum().item()) if masks is not None else 0
        print(f"  {snap.prompts[i.prompt_idx]!r}: score {i.score:.2f} box {np.round(i.box).astype(int).tolist()} "
              f"mask px {area}")  # fmt: skip
    if args.out and masks is not None:
        color = np.zeros_like(rgb)
        conf = np.zeros((h, w), np.float32)
        for k in range(len(insts) - 1, -1, -1):
            m = masks[k].cpu().numpy() > 0.5
            color[m] = PALETTE[insts[k].prompt_idx % len(PALETTE)]
            conf[m] = 1.0
        import cv2

        cv2.imwrite(args.out, cv2.cvtColor(blend_overlay(rgb, color, conf, 0.55), cv2.COLOR_RGB2BGR))
        print("overlay ->", args.out)


if __name__ == "__main__":
    main()
