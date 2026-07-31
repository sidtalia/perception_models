"""CLIPSeg + NanoSAM backends for HOUND perception.

Engines/ONNX: ``perception_models/data/``.
Rebuild scripts: ``perception_models/scripts/``.
Dora/ROS nodes: ``hound_core``.
"""

__all__ = [
    "ClipSegParts",
    "PromptLayout",
    "prompt_layout_from_config",
    "NanoSamEncoder",
    "NanoSamDecoder",
]


def __getattr__(name: str):
    if name in ("ClipSegParts", "PromptLayout", "prompt_layout_from_config"):
        from perception_models.clipseg.shared import (
            ClipSegParts,
            PromptLayout,
            prompt_layout_from_config,
        )

        return {
            "ClipSegParts": ClipSegParts,
            "PromptLayout": PromptLayout,
            "prompt_layout_from_config": prompt_layout_from_config,
        }[name]
    if name in ("NanoSamEncoder", "NanoSamDecoder"):
        from perception_models.nanosam_trt.refiner import (
            NanoSamDecoder,
            NanoSamEncoder,
        )

        return {
            "NanoSamEncoder": NanoSamEncoder,
            "NanoSamDecoder": NanoSamDecoder,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
