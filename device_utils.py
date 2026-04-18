"""
Device abstraction for TPU / CUDA / MPS / CPU.

The pipeline is primarily developed for CUDA + MPS. This module adds a thin
compatibility layer so that the same code paths also run on Google Cloud TPUs
via `torch_xla`, without having to sprinkle `is_xla_available()` checks across
callers.

Priority order for auto-selection: XLA > CUDA > MPS > CPU.

Design constraints:
 - `torch.cuda.is_available()` returns False on TPU, so CUDA-only branches are
   skipped without changes.
 - `device_map="auto"` from accelerate does not support XLA; use
   `load_model_for_device()` instead of `from_pretrained(..., device_map="auto")`.
 - `torch.cuda.empty_cache()` has no direct XLA analogue; `empty_cache()` here
   is a no-op under XLA but triggers `xm.mark_step()` so that pending ops are
   flushed (which is the closest XLA equivalent).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import torch_xla.core.xla_model as xm  # type: ignore
    _XLA_AVAILABLE = True
except Exception:
    xm = None  # type: ignore
    _XLA_AVAILABLE = False


def is_xla_available() -> bool:
    """True iff torch_xla is importable."""
    return _XLA_AVAILABLE


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device() -> torch.device:
    """Return the best available device as a torch.device.

    Priority: XLA > CUDA > MPS > CPU.
    """
    if is_xla_available():
        return xm.xla_device()  # type: ignore[attr-defined]
    if is_cuda_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_str() -> str:
    """String form of `get_device()`; useful for APIs expecting a string."""
    return str(get_device())


def empty_cache() -> None:
    """Device-appropriate cache clearing.

    - CUDA: `torch.cuda.empty_cache()`
    - MPS:  `torch.mps.empty_cache()` (when available)
    - XLA:  `xm.mark_step()` — flushes pending graph; XLA has no user-visible
            cache, but mark_step is the closest equivalent for releasing
            intermediate buffers between stages.
    - CPU:  no-op.
    """
    if is_xla_available():
        try:
            xm.mark_step()  # type: ignore[attr-defined]
        except Exception:
            pass
        return
    if is_cuda_available():
        torch.cuda.empty_cache()
        return
    if is_mps_available() and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def mark_step() -> None:
    """XLA-only step marker. No-op elsewhere.

    Call between distinct compute phases on TPU to avoid accumulating a single
    enormous graph and to release intermediate buffers.
    """
    if is_xla_available():
        try:
            xm.mark_step()  # type: ignore[attr-defined]
        except Exception:
            pass


def device_map_for_loading(device: Optional[torch.device | str] = None):
    """Return the appropriate `device_map` kwarg for `from_pretrained`.

    - On XLA: returns None (caller should load to CPU then `.to(device)`).
      accelerate's device_map="auto" does not understand XLA devices.
    - Elsewhere: returns "auto" to preserve existing behavior.
    """
    if is_xla_available():
        return None
    return "auto"


def load_model_for_device(
    model_cls,
    model_path: str,
    *,
    torch_dtype=torch.bfloat16,
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = None,
    **extra_kwargs,
):
    """Load an HF causal LM onto the active device, handling the XLA case.

    On CUDA/MPS/CPU this is equivalent to the original pipeline's
    `from_pretrained(..., device_map="auto")`. On XLA, it loads weights onto
    CPU first (device_map left unset) and then moves to `xm.xla_device()`,
    since accelerate does not support XLA device placement.

    `model_cls` is typically `transformers.AutoModelForCausalLM`.
    """
    kwargs = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    kwargs.update(extra_kwargs)

    dmap = device_map_for_loading()
    if dmap is not None:
        kwargs["device_map"] = dmap

    model = model_cls.from_pretrained(model_path, **kwargs)

    if is_xla_available():
        model = model.to(get_device())

    return model
