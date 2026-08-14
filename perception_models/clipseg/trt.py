"""TensorRT CLIPSeg vision encoder + FiLM decoder (batched engines only)."""

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


def load_clipseg_film_engine(
    path: str,
    *,
    hs_name: str = "stacked_hs",
    cond_name: str = "cond",
    output_name: str = "probs",
):
    """Load a TRT engine built by scripts/build_clipseg_film_engine.sh."""
    import tensorrt as trt
    from torch2trt import TRTModule

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"CLIPSeg FiLM engine not found: {p}")

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with p.open("rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    return TRTModule(
        engine=engine,
        input_names=[hs_name, cond_name],
        output_names=[output_name],
    )


def _engine_tensor_shape(engine_module, name: str) -> tuple[int, ...] | None:
    eng = getattr(engine_module, "engine", None)
    if eng is None:
        return None
    try:
        shape = tuple(eng.get_tensor_shape(name))
    except Exception:
        return None
    return shape if shape else None


def engine_input_batch(engine_module) -> int | None:
    """Fixed input batch dim for ``pixel_values``, or None if dynamic/unknown."""
    shape = _engine_tensor_shape(engine_module, "pixel_values")
    if not shape:
        return None
    b = int(shape[0])
    return b if b > 0 else None


def film_engine_shapes(engine_module) -> tuple[int | None, int | None]:
    """Return ``(batch_n, num_prompts)`` from FiLM engine bindings (or None)."""
    hs = _engine_tensor_shape(engine_module, "stacked_hs")
    cond = _engine_tensor_shape(engine_module, "cond")
    n = int(hs[0]) if hs and int(hs[0]) > 0 else None
    p = int(cond[0]) if cond and int(cond[0]) > 0 else None
    return n, p


class TrtFilmDecoder:
    """Callable matching ``dec(stacked_hs, cond) -> [N,P,H,W]`` (fp16 CUDA)."""

    def __init__(self, engine_module):
        self.engine = engine_module
        self._batch_n, self._num_prompts = film_engine_shapes(engine_module)

    def __call__(self, stacked_hs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return trt_decode_film(self.engine, stacked_hs, cond)


@torch.inference_mode()
def trt_encode_hs(engine, pixel_values: torch.Tensor) -> torch.Tensor:
    """Run TRT vision engine → hidden states ``[N,L,S,D]`` float32 CUDA (cloned).

    ``pixel_values``: normalized CLIP pixels ``[N,3,res,res]``.
    Engine must accept this N (fixed batch-N or dynamic). No per-image loop.
    """
    x = pixel_values.float().contiguous()
    n = int(x.shape[0])
    eng_batch = engine_input_batch(engine)
    if eng_batch is not None and eng_batch != n:
        raise RuntimeError(
            f"CLIPSeg TRT engine batch={eng_batch} but got N={n}; "
            "use a matching batched engine (no serial fallback)"
        )
    out = engine(x)
    hs = out[0] if isinstance(out, (tuple, list)) else out
    hs = hs.float()
    if hs.dim() == 3:
        hs = hs.unsqueeze(0)
    if int(hs.shape[0]) != n:
        raise RuntimeError(
            f"CLIPSeg TRT output batch {hs.shape[0]} != input N={n}"
        )
    return hs.contiguous().clone()


@torch.inference_mode()
def trt_decode_film(
    engine, stacked_hs: torch.Tensor, cond: torch.Tensor
) -> torch.Tensor:
    """Run TRT FiLM decoder → prompt probs ``[N,P,H,W]`` fp16 CUDA (cloned)."""
    hs = stacked_hs.float().contiguous()
    if hs.dim() == 3:
        hs = hs.unsqueeze(0)
    if hs.dim() != 4:
        raise ValueError(f"stacked_hs expected [N,L,S,D], got {tuple(hs.shape)}")
    c = cond.float().contiguous()
    if c.dim() != 2:
        raise ValueError(f"cond expected [P,C], got {tuple(c.shape)}")
    n = int(hs.shape[0])
    p = int(c.shape[0])
    eng_n, eng_p = film_engine_shapes(engine)
    if eng_n is not None and eng_n != n:
        raise RuntimeError(
            f"CLIPSeg FiLM TRT batch={eng_n} but got N={n}; "
            "use a matching batched engine (no serial fallback)"
        )
    # Allow fewer live prompts than a fixed-P engine (zero-pad, then slice).
    if eng_p is not None and p != eng_p:
        if p > eng_p:
            raise RuntimeError(
                f"CLIPSeg FiLM TRT prompts={eng_p} but got P={p}; "
                "rebuild engine for this prompt count"
            )
        c = torch.cat(
            [c, c.new_zeros((eng_p - p, c.shape[1]), dtype=c.dtype, device=c.device)],
            dim=0,
        )
    out = engine(hs, c)
    probs = out[0] if isinstance(out, (tuple, list)) else out
    if probs.dim() == 3:
        probs = probs.unsqueeze(0)
    if eng_p is not None and int(probs.shape[1]) == eng_p and p < eng_p:
        probs = probs[:, :p]
    if int(probs.shape[0]) != n or int(probs.shape[1]) != p:
        raise RuntimeError(
            f"CLIPSeg FiLM TRT output {tuple(probs.shape)} != N={n} P={p}"
        )
    return probs.to(dtype=torch.float16).contiguous().clone()
