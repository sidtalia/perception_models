"""Optional TensorRT CLIPSeg vision encoder (same hs contract as torch path)."""

from __future__ import annotations

from pathlib import Path

import torch


def load_clipseg_vision_engine(
    path: str,
    *,
    input_name: str = "pixel_values",
    output_name: str = "hidden_states",
):
    """Load a TRT engine built by scripts/build_clipseg_vision_engine.sh."""
    import tensorrt as trt
    from torch2trt import TRTModule

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"CLIPSeg vision engine not found: {p}")

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with p.open("rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    return TRTModule(
        engine=engine,
        input_names=[input_name],
        output_names=[output_name],
    )


@torch.inference_mode()
def trt_encode_hs(engine, pixel_values: torch.Tensor) -> torch.Tensor:
    """Run TRT vision engine → hidden states [L,S,D] float32 CUDA (cloned).

    ``pixel_values``: normalized CLIP pixels [1,3,res,res] (fp16/fp32).
    """
    x = pixel_values.float().contiguous()
    out = engine(x)
    hs = out[0] if isinstance(out, (tuple, list)) else out
    hs = hs.float()
    if hs.dim() == 4 and hs.shape[0] == 1:
        hs = hs.squeeze(0)
    return hs.contiguous().clone()
