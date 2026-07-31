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

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_seg_config(path: str | None = None) -> dict[str, Any]:
    """Load segmentation JSON written by launch (HOUND_SEG_CONFIG)."""
    cfg_path = path or os.environ.get("HOUND_SEG_CONFIG", "")
    if not cfg_path:
        raise RuntimeError("HOUND_SEG_CONFIG is not set")
    with open(cfg_path, "r", encoding="utf-8") as f:
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

        tok = self.proc.tokenizer(
            prompts.full_prompts, return_tensors="pt", padding=True
        ).to(device)
        with torch.inference_mode():
            self.cond = self.model.get_conditional_embeddings(
                batch_size=len(prompts.full_prompts),
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
            )

        self._encode_mod: nn.Module | None = None
        self._decode_mod: nn.Module | None = None
        self.compile_mode = compile_mode
        self._logged_map_shape = False

    def _info(self, msg: str):
        if self._log:
            self._log(msg)

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
        t = (
            torch.from_numpy(rgb_uint8)
            .to(self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div_(255.0)
        )
        pix = F.interpolate(
            t, size=(self.res, self.res), mode="bilinear", align_corners=False
        )
        return (pix - self.mean) / self.std

    @torch.inference_mode()
    def encode(self, rgb_uint8: np.ndarray, enc: nn.Module) -> torch.Tensor:
        """-> stacked hidden states [L, S, D]."""
        pix = self.preprocess_rgb(rgb_uint8)
        with torch.autocast("cuda", dtype=torch.float16):
            hs = enc(pix)
        return hs.float().clone()

    @torch.inference_mode()
    def decode_probs(
        self,
        stacked_hs: torch.Tensor,
        dec: nn.Module,
        *,
        height: int,
        width: int,
        aggregate: str,
    ) -> torch.Tensor:
        """Decode + aggregate + upsample -> group probs [G,H,W] at full resolution.

        ``stacked_hs``: [L,S,D] from the vision tower.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            sub = dec(stacked_hs, self.cond).float()  # [P,res,res]
        if not self._logged_map_shape:
            eff = self.res // 16
            self._info(
                f"CLIPSeg FiLM map {tuple(sub.shape)} (~{eff}x{eff} token grid)"
            )
            self._logged_map_shape = True
        groups = self.aggregate_groups(sub, aggregate)
        if self.prompts.suppress_idx:
            sky = sub[self.prompts.suppress_idx].amax(dim=0)
            groups = (groups - sky.unsqueeze(0)).clamp_(0.0, 1.0)
        probs = F.interpolate(
            groups.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].clamp_(0.0, 1.0)
        return probs

    def aggregate_groups(self, sub: torch.Tensor, aggregate: str) -> torch.Tensor:
        """Per-sub-prompt maps [P,H,W] -> per-group maps [n_groups,H,W]."""
        out = sub.new_zeros((self.prompts.n_groups, sub.shape[1], sub.shape[2]))
        agg = (aggregate or "max").lower()
        for gi, idx in enumerate(self.prompts.group_index):
            if not idx:
                continue
            s = sub[idx]
            if agg == "mean":
                out[gi] = s.mean(dim=0)
            elif agg == "sum":
                out[gi] = s.sum(dim=0).clamp_(0.0, 1.0)
            elif agg == "noisy_or":
                out[gi] = 1.0 - (1.0 - s).prod(dim=0)
            else:
                out[gi] = s.amax(dim=0)
        return out

    def _make_encoder_eager(self) -> nn.Module:
        outer = self

        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = outer.model
                self.extract = list(outer.extract)

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                # pixel_values: [1,3,res,res] -> [L,S,D]
                vo = self.model.clip.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
                layers = [
                    vo.hidden_states[i + 1].contiguous()  # [1,S,D]
                    for i in self.extract
                ]
                return torch.stack(layers, dim=1)[0]  # [L,S,D]

        return _Enc().eval()

    def _make_decoder_eager(self) -> nn.Module:
        outer = self

        class _Dec(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = outer.model

            def forward(self, stacked_hs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
                # stacked_hs: [L,S,D]; batch = n_prompts.
                n = cond.shape[0]
                hs = tuple(
                    stacked_hs[i].unsqueeze(0).expand(n, -1, -1).contiguous()
                    for i in range(stacked_hs.shape[0])
                )
                return self.model.decoder(hs, cond).logits.sigmoid()

        return _Dec().eval()

    def warmup_encoder(self, enc: nn.Module) -> nn.Module:
        dummy = torch.zeros(1, 3, self.res, self.res, device=self.device)

        def run(mod: nn.Module):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for _ in range(3):
                    _ = mod(dummy)
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

    def warmup_decoder(self, dec: nn.Module) -> nn.Module:
        enc = self._encode_mod if self._encode_mod is not None else self._make_encoder_eager()
        dummy = torch.zeros(1, 3, self.res, self.res, device=self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            hs = enc(dummy).float()

        def run(mod: nn.Module):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for _ in range(3):
                    _ = mod(hs, self.cond)
            torch.cuda.synchronize()

        try:
            run(dec)
        except Exception as e:  # noqa: BLE001
            self._info(f"decoder compile/warmup failed ({e}); eager fallback")
            dec = self._make_decoder_eager()
            self._decode_mod = dec
            run(dec)
        self._info("CLIPSeg decoder warmup/compile complete.")
        return dec
