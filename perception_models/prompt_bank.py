"""Text-prompt embeddings computed off the hot path.

The FiLM decoder only needs a ``cond`` tensor ``[P, C]``. Computing it means
running the tokenizer and the CLIP text tower, which takes milliseconds and
stalls the frame loop. :class:`PromptBank` does that work on a background
thread and publishes an immutable :class:`PromptSet` snapshot by swapping a
single reference under a lock. The frame loop calls :meth:`PromptBank.current`
once per frame and uses that snapshot for the whole frame, so the prompts never
change mid-decode.

Updates coalesce: if prompts arrive faster than they can be embedded, only the
newest request is processed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptSet:
    """Immutable prompt snapshot used for one or more frames."""

    version: int
    prompts: tuple[str, ...]
    cond: Any  # torch.Tensor [P, C] on the decode device

    @property
    def n(self) -> int:
        return len(self.prompts)


EmbedFn = Callable[[Sequence[str]], Any]


class PromptBank:
    """Background prompt embedder with atomic snapshot swap.

    Args:
        embed_fn: ``prompts -> cond`` (runs on the worker thread; should use its
            own CUDA stream and synchronize it before returning).
        max_prompts: hard cap (the FiLM TRT engine has a fixed prompt slot count).
        on_update: optional callback ``(PromptSet) -> None`` after each swap.
        on_error: optional callback ``(prompts, exc) -> None`` if embedding fails;
            the previous snapshot stays active.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        *,
        max_prompts: int | None = None,
        on_update: Callable[[PromptSet], None] | None = None,
        on_error: Callable[[Sequence[str], BaseException], None] | None = None,
    ) -> None:
        self._embed = embed_fn
        self._max = max_prompts
        self._on_update = on_update
        self._on_error = on_error
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._current: PromptSet | None = None
        self._requested: tuple[str, ...] | None = None
        self._version = 0
        self._stop = False
        self._busy = False
        self._thread = threading.Thread(target=self._run, name="prompt_bank", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ API
    def current(self) -> PromptSet | None:
        """Latest ready snapshot (``None`` until the first embedding finishes)."""
        with self._lock:
            return self._current

    def request(self, prompts: Sequence[str]) -> bool:
        """Queue a prompt update. Returns ``False`` if rejected (empty / over capacity / unchanged)."""
        clean = tuple(str(p).strip() for p in prompts if str(p).strip())
        if not clean:
            return False
        if self._max is not None and len(clean) > self._max:
            raise ValueError(f"{len(clean)} prompts exceed the engine's {self._max} prompt slots")
        with self._wake:
            pending = self._requested
            active = self._current.prompts if self._current else None
            if clean == pending or (pending is None and clean == active and not self._busy):
                return False
            self._requested = clean
            self._wake.notify()
        return True

    def set_blocking(self, prompts: Sequence[str], timeout: float = 120.0) -> PromptSet:
        """Request and wait until that exact prompt set is active (startup path)."""
        target = tuple(str(p).strip() for p in prompts if str(p).strip())
        self.request(target)

        def ready() -> bool:
            cur = self._current
            return cur is not None and cur.prompts == target and self._requested is None

        with self._wake:
            ok = self._wake.wait_for(lambda: ready() or self._stop, timeout=timeout)
        if not ok or self._current is None or self._current.prompts != target:
            raise TimeoutError(f"prompt embedding for {list(target)} did not finish in {timeout}s")
        return self._current

    def close(self) -> None:
        with self._wake:
            self._stop = True
            self._wake.notify_all()
        self._thread.join(timeout=5.0)

    # --------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            with self._wake:
                self._wake.wait_for(lambda: self._stop or self._requested is not None)
                if self._stop:
                    return
                prompts = self._requested
                self._requested = None
                self._busy = True
            assert prompts is not None
            try:
                cond = self._embed(prompts)
            except BaseException as exc:  # noqa: BLE001 - keep the worker alive
                with self._wake:
                    self._busy = False
                    self._wake.notify_all()
                if self._on_error:
                    self._on_error(prompts, exc)
                continue
            with self._wake:
                self._version += 1
                snap = PromptSet(self._version, prompts, cond)
                self._current = snap
                self._busy = False
                self._wake.notify_all()
            if self._on_update:
                self._on_update(snap)
