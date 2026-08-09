"""CLIPSeg encode / decode (torch + optional TRT vision)."""

from __future__ import annotations

__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "ClipSegParts",
    "PromptLayout",
    "load_seg_config",
    "prompt_layout_from_config",
    "load_clipseg_vision_engine",
    "load_clipseg_film_engine",
    "TrtFilmDecoder",
    "trt_encode_hs",
    "trt_decode_film",
]


def __getattr__(name: str):
    if name in (
        "CLIP_MEAN",
        "CLIP_STD",
        "ClipSegParts",
        "PromptLayout",
        "load_seg_config",
        "prompt_layout_from_config",
    ):
        from perception_models.clipseg import shared as _shared

        return getattr(_shared, name)
    if name in (
        "load_clipseg_vision_engine",
        "load_clipseg_film_engine",
        "TrtFilmDecoder",
        "trt_encode_hs",
        "trt_decode_film",
    ):
        from perception_models.clipseg import trt as _trt

        return getattr(_trt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
