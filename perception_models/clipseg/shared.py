"""Shared CLIPSeg encode / FiLM-decode helpers.

One vision encode, then batched FiLM decoder across all prompts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_seg_config(path: str | None = None) -> dict[str, Any]:
    """Load the segmentation config (JSON or YAML).

    Resolution order: ``path`` argument, ``--config <file>`` on the command line,
    then the ``SEG_CONFIG`` / ``HOUND_SEG_CONFIG`` environment variables.
    """
    import sys

    cfg_path = path
    if not cfg_path and "--config" in sys.argv:
        i = sys.argv.index("--config")
        if i + 1 < len(sys.argv):
            cfg_path = sys.argv[i + 1]
    cfg_path = cfg_path or os.environ.get("SEG_CONFIG") or os.environ.get("HOUND_SEG_CONFIG", "")
    if not cfg_path:
        raise RuntimeError("no segmentation config: pass --config <file> or set SEG_CONFIG")
    with open(cfg_path, "r", encoding="utf-8") as f:
        if cfg_path.endswith((".yaml", ".yml")):
            import yaml

            return dict(yaml.safe_load(f) or {})
        return json.load(f)


@dataclass
class PromptLayout:
    subprompts: list[str]
    group_names: list[str]
    group_ids: list[int]
    suppress_prompts: list[str]
    full_prompts: list[str] = field(init=False)
    suppress_idx: list[int] = field(init=False)
    n_groups: int = field(init=False)
    group_index: list[list[int]] = field(init=False)

    def __post_init__(self):
        if not self.group_names or len(self.group_ids) != len(self.subprompts):
            self.group_names = list(self.subprompts)
            self.group_ids = list(range(len(self.subprompts)))
        self.n_groups = len(self.group_names)
        self.group_index = [
            [i for i, gid in enumerate(self.group_ids) if gid == gi]
            for gi in range(self.n_groups)
        ]
        self.full_prompts = list(self.subprompts) + list(self.suppress_prompts)
        self.suppress_idx = list(range(len(self.subprompts), len(self.full_prompts)))


def prompt_layout_from_config(cfg: dict[str, Any]) -> PromptLayout:
    return PromptLayout(
        subprompts=[str(p) for p in cfg.get("prompts", ["a person"])],
        group_names=[str(x) for x in cfg.get("group_names", []) if str(x) != ""],
        group_ids=[int(x) for x in cfg.get("group_ids", [])],
        suppress_prompts=[
            str(x) for x in cfg.get("suppress_prompts", []) if str(x) != ""
        ],
    )


class ClipSegParts:
    """Vision encoder + FiLM decoder with precomputed text conditionals."""

    def __init__(
        self,
        model_name: str,
        prompts: PromptLayout,
        *,
        input_res: int = 224,
        compile_mode: str = "reduce-overhead",
        device: str = "cuda",
        logger=None,
    ):
        self.device = device
        self.res = int(input_res)
        self.prompts = prompts
        self._log = logger
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

        if self._log:
            self._log(f"Loading CLIPSeg model '{model_name}' ...")
        self.proc = CLIPSegProcessor.from_pretrained(model_name)
        self.model = (
            CLIPSegForImageSegmentation.from_pretrained(model_name).to(device).eval()
        )
        self.extract = list(self.model.extract_layers)

        self.cond = self.embed_prompts(prompts.full_prompts)

        self._encode_mod: nn.Module | None = None
        self._decode_mod: nn.Module | None = None
        self.compile_mode = compile_mode
        self._logged_map_shape = False
        # Reused pinned NHWC staging for preprocess_rgb_batch (avoids per-frame alloc).
        self._pin_nhwc: torch.Tensor | None = None

    @torch.inference_mode()
    def embed_prompts(
        self, prompts, *, stream: torch.cuda.Stream | None = None
    ) -> torch.Tensor:
        """Tokenize + CLIP text tower → FiLM conditionals ``[P, C]`` (fp32, on device).

        Safe to call from a background thread: pass a dedicated ``stream`` so the
        text tower does not serialize behind the frame loop's kernels. The stream
        is synchronized before returning, so the tensor is ready to use anywhere.
        """
        prompts = [str(p) for p in prompts]
        if not prompts:
            raise ValueError("at least one prompt is required")
        tok = self.proc.tokenizer(prompts, return_tensors="pt", padding=True)
        ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
        with ctx:
            ids = tok.input_ids.to(self.device, non_blocking=True)
            mask = tok.attention_mask.to(self.device, non_blocking=True)
            cond = self.model.get_conditional_embeddings(
                batch_size=len(prompts), input_ids=ids, attention_mask=mask
            ).float().contiguous()
        if stream is not None:
            stream.synchronize()
        return cond

    def _info(self, msg: str):
        if self._log:
            self._log(msg)

    def _ensure_pin_nhwc(self, n: int, h: int, w: int) -> torch.Tensor:
        pin = self._pin_nhwc
        if (
            pin is None
            or pin.shape[0] < n
            or pin.shape[1] != h
            or pin.shape[2] != w
        ):
            pin = torch.empty((n, h, w, 3), dtype=torch.uint8, pin_memory=True)
            self._pin_nhwc = pin
        return pin[:n]

    def build_encoder(self) -> nn.Module:
        eager = self._make_encoder_eager()
        self._encode_mod = self._maybe_compile(eager, "encoder")
        return self._encode_mod

    def build_decoder(self) -> nn.Module:
        eager = self._make_decoder_eager()
        self._decode_mod = self._maybe_compile(eager, "decoder")
        return self._decode_mod

    def _maybe_compile(self, mod: nn.Module, name: str) -> nn.Module:
        mode = (self.compile_mode or "none").lower()
        if mode == "none":
            return mod
        self._info(f"torch.compile({name}, mode='{self.compile_mode}') ...")
        compiled = torch.compile(mod, mode=self.compile_mode, dynamic=False)
        return compiled

    @torch.inference_mode()
    def preprocess_rgb(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 -> normalized [1,3,res,res] float on device."""
        return self.preprocess_rgb_batch([rgb_uint8])

    @torch.inference_mode()
    def preprocess_rgb_batch(self, rgbs: list) -> torch.Tensor:
        """List of HxWx3 uint8 -> normalized [N,3,res,res] float on device.

        One host stack into pinned NHWC → single H2D → NCHW resize/normalize.
        """
        if not rgbs:
            raise ValueError("empty rgb batch")
        n = len(rgbs)
        h, w = int(rgbs[0].shape[0]), int(rgbs[0].shape[1])
        pin = self._ensure_pin_nhwc(n, h, w)
        host = pin.numpy()
        for i, rgb in enumerate(rgbs):
            np.copyto(host[i], np.ascontiguousarray(rgb, dtype=np.uint8))
        pix = (
            pin.to(device=self.device, non_blocking=True)
            .permute(0, 3, 1, 2)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        pix = F.interpolate(
            pix, size=(self.res, self.res), mode="bilinear", align_corners=False
        )
        return (pix - self.mean) / self.std

    @torch.inference_mode()
    def encode(self, rgb_uint8: np.ndarray, enc: nn.Module) -> torch.Tensor:
        """-> stacked hidden states [L, S, D]."""
        hs = self.encode_batch([rgb_uint8], enc)
        return hs[0]

    @torch.inference_mode()
    def encode_batch(self, rgbs: list, enc: nn.Module) -> torch.Tensor:
        """-> stacked hidden states [N, L, S, D]."""
        pix = self.preprocess_rgb_batch(rgbs)
        with torch.autocast("cuda", dtype=torch.float16):
            hs = enc(pix)
        if hs.dim() == 3:
            hs = hs.unsqueeze(0)
        return hs.float().clone()

    @torch.inference_mode()
    def decode_probs(
        self,
        stacked_hs: torch.Tensor,
        dec: nn.Module,
        *,
        height: int | None = None,
        width: int | None = None,
        aggregate: str,
    ) -> torch.Tensor:
        """Decode + aggregate (+ optional upsample) → group probs ``[G,H,W]``.

        ``stacked_hs``: [L,S,D] from the vision tower.
        When ``height``/``width`` are omitted, returns native FiLM resolution.
        """
        out = self.decode_probs_batch(
            stacked_hs.unsqueeze(0) if stacked_hs.dim() == 3 else stacked_hs,
            dec,
            height=height,
            width=width,
            aggregate=aggregate,
        )
        return out[0]

    @torch.inference_mode()
    def decode_probs_batch(
        self,
        stacked_hs: torch.Tensor,
        dec: nn.Module,
        *,
        aggregate: str,
        height: int | None = None,
        width: int | None = None,
        cond: torch.Tensor | None = None,
        layout: PromptLayout | None = None,
    ) -> torch.Tensor:
        """Batched FiLM across images → group probs ``[N,G,H,W]`` (fp16).

        ``cond`` / ``layout`` override the prompts fixed at construction (used
        with a :class:`~perception_models.prompt_bank.PromptBank` snapshot so
        prompts can change at runtime without touching shared state).

        ``stacked_hs``: ``[N,L,S,D]`` (or ``[L,S,D]``, treated as N=1).
        One decoder enqueue of size ``N * n_prompts`` instead of N serial decodes.
        Stays in fp16 through aggregate / optional upsample; cast at host boundary.
        Omit ``height``/``width`` to keep native FiLM map size (for boxes/CC).
        """
        if stacked_hs.dim() == 3:
            stacked_hs = stacked_hs.unsqueeze(0)
        if stacked_hs.dim() != 4:
            raise ValueError(
                f"stacked_hs expected [N,L,S,D] or [L,S,D], got {tuple(stacked_hs.shape)}"
            )

        cond = self.cond if cond is None else cond
        layout = self.prompts if layout is None else layout
        with torch.autocast("cuda", dtype=torch.float16):
            sub = dec(stacked_hs, cond)  # [N,P,res,res]
            if sub.dtype != torch.float16:
                sub = sub.to(dtype=torch.float16)
            if not self._logged_map_shape:
                self._info(f"CLIPSeg FiLM map {tuple(sub.shape)} dtype={sub.dtype}")
                self._logged_map_shape = True

            groups = self.aggregate_groups_batch(sub, aggregate, layout)  # [N,G,h,w]
            if layout.suppress_idx:
                sky = sub[:, layout.suppress_idx].amax(dim=1)  # [N,h,w]
                groups = (groups - sky.unsqueeze(1)).clamp_(0.0, 1.0)
            if height is None or width is None:
                return groups.clamp_(0.0, 1.0)
            return F.interpolate(
                groups,
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
            ).clamp_(0.0, 1.0)

    def aggregate_groups(self, sub: torch.Tensor, aggregate: str) -> torch.Tensor:
        """Per-sub-prompt maps [P,H,W] -> per-group maps [n_groups,H,W]."""
        return self.aggregate_groups_batch(sub.unsqueeze(0), aggregate)[0]

    def aggregate_groups_batch(
        self, sub: torch.Tensor, aggregate: str, layout: PromptLayout | None = None
    ) -> torch.Tensor:
        """Batched [N,P,H,W] -> [N,n_groups,H,W]."""
        if sub.dim() != 4:
            raise ValueError(f"expected [N,P,H,W], got {tuple(sub.shape)}")
        layout = self.prompts if layout is None else layout
        n, _p, hh, ww = sub.shape
        out = sub.new_zeros((n, layout.n_groups, hh, ww))
        agg = (aggregate or "max").lower()
        for gi, idx in enumerate(layout.group_index):
            if not idx:
                continue
            s = sub[:, idx]  # [N,k,H,W]
            if agg == "mean":
                out[:, gi] = s.mean(dim=1)
            elif agg == "sum":
                out[:, gi] = s.sum(dim=1).clamp_(0.0, 1.0)
            elif agg == "noisy_or":
                out[:, gi] = 1.0 - (1.0 - s).prod(dim=1)
            else:
                out[:, gi] = s.amax(dim=1)
        return out

    def _make_encoder_eager(self) -> nn.Module:
        outer = self

        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = outer.model
                self.extract = list(outer.extract)

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                # pixel_values: [N,3,res,res] -> [N,L,S,D] (N=1 squeezable by caller)
                vo = self.model.clip.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
                layers = [
                    vo.hidden_states[i + 1].contiguous()  # [N,S,D]
                    for i in self.extract
                ]
                # stack on layer dim → [N,L,S,D]
                return torch.stack(layers, dim=1)

        return _Enc().eval()

    def _make_decoder_eager(self) -> nn.Module:
        outer = self

        class _Dec(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = outer.model

            def forward(self, stacked_hs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
                """FiLM decode.

                * ``stacked_hs`` ``[L,S,D]`` → logits ``[P,H,W]`` (prompt-batched)
                * ``stacked_hs`` ``[N,L,S,D]`` → logits ``[N,P,H,W]`` (image×prompt)
                """
                p = int(cond.shape[0])
                if stacked_hs.dim() == 3:
                    hs = tuple(
                        stacked_hs[i].unsqueeze(0).expand(p, -1, -1).contiguous()
                        for i in range(stacked_hs.shape[0])
                    )
                    return self.model.decoder(hs, cond).logits.sigmoid()

                if stacked_hs.dim() != 4:
                    raise ValueError(
                        f"decoder expected [L,S,D] or [N,L,S,D], got {tuple(stacked_hs.shape)}"
                    )
                n, l, s, d = stacked_hs.shape
                hs = tuple(
                    stacked_hs[:, i]
                    .unsqueeze(1)
                    .expand(-1, p, -1, -1)
                    .reshape(n * p, s, d)
                    .contiguous()
                    for i in range(l)
                )
                cond_b = (
                    cond.unsqueeze(0).expand(n, -1, -1).reshape(n * p, -1).contiguous()
                )
                out = self.model.decoder(hs, cond_b).logits.sigmoid()  # [N*P,H,W]
                return out.view(n, p, out.shape[-2], out.shape[-1])

        return _Dec().eval()

    def warmup_encoder(self, enc: nn.Module) -> nn.Module:
        dummy = torch.zeros(1, 3, self.res, self.res, device=self.device)

        def run(mod: nn.Module):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for _ in range(3):
                    out = mod(dummy)
                    if out.dim() == 4 and out.shape[0] == 1:
                        out = out[0]
                    _ = out
            torch.cuda.synchronize()

        try:
            run(enc)
        except Exception as e:  # noqa: BLE001
            self._info(f"encoder compile/warmup failed ({e}); eager fallback")
            enc = self._make_encoder_eager()
            self._encode_mod = enc
            run(enc)
        self._info("CLIPSeg encoder warmup/compile complete.")
        return enc

    def warmup_decoder(
        self, dec: nn.Module, *, batch_n: int = 1
    ) -> nn.Module:
        enc = self._encode_mod if self._encode_mod is not None else self._make_encoder_eager()
        bn = max(1, int(batch_n))
        dummy = torch.zeros(bn, 3, self.res, self.res, device=self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            hs = enc(dummy).float()
            if hs.dim() == 3:
                hs = hs.unsqueeze(0)

        def run(mod: nn.Module):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for _ in range(3):
                    # Warm both contracts (compile caches shapes).
                    _ = mod(hs[0], self.cond)
                    _ = mod(hs if bn > 1 else hs[:1], self.cond)
            torch.cuda.synchronize()

        try:
            run(dec)
        except Exception as e:  # noqa: BLE001
            self._info(f"decoder compile/warmup failed ({e}); eager fallback")
            dec = self._make_decoder_eager()
            self._decode_mod = dec
            run(dec)
        self._info(
            f"CLIPSeg decoder warmup/compile complete (batch_n={bn})."
        )
        return dec
