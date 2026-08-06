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
    """Run TRT vision engine → hidden states float32 CUDA (cloned).

    ``pixel_values``: normalized CLIP pixels [N,3,res,res] (fp16/fp32).
    Returns [L,S,D] when N==1 (legacy), else [N,L,S,D].

    Engines built for batch=1 will process N via a Python loop.
    Engines built for batch=N (see export --batch) run in one shot.
    """
    x = pixel_values.float().contiguous()
    n = int(x.shape[0])
    if n == 1:
        out = engine(x)
        hs = out[0] if isinstance(out, (tuple, list)) else out
        hs = hs.float()
        if hs.dim() == 4 and hs.shape[0] == 1:
            hs = hs.squeeze(0)
        return hs.contiguous().clone()

    # Try native batch; fall back to per-image if engine is fixed batch=1.
    try:
        out = engine(x)
        hs = out[0] if isinstance(out, (tuple, list)) else out
        hs = hs.float()
        if hs.dim() == 3:
            hs = hs.unsqueeze(0)
        if hs.shape[0] == n:
            return hs.contiguous().clone()
    except Exception:
        pass

    parts = []
    for i in range(n):
        out = engine(x[i : i + 1])
        hs = out[0] if isinstance(out, (tuple, list)) else out
        hs = hs.float()
        if hs.dim() == 4 and hs.shape[0] == 1:
            hs = hs.squeeze(0)
        parts.append(hs.contiguous())
    return torch.stack(parts, dim=0).clone()
