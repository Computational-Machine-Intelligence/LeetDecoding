"""CUDA event-based nested profiling timer for Transformer attention breakdown.

Core design:
- add_module(module, tag, name) → hooks entire module via forward pre/post hooks.
- range(tag, name) → context manager for manual instrumentation (e.g., attention core).
- Inclusive / exclusive relationship is preserved in the raw records.
  Callers must compute non-overlapping breakdown (e.g., attn_non_core = attn_module - attn_core).
"""

from __future__ import annotations

import gc
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch


class CUDABlockTimer:
    """Records CUDA event ranges for per-layer and per-component timing.

    Usage::

        timer = CUDABlockTimer()
        for layer in model.layers:
            timer.add_module(layer.self_attn,  tag="attn_module", name=f"L{idx}.attn")
            timer.add_module(layer.mlp,        tag="mlp",         name=f"L{idx}.mlp")
        timer.enabled = True
        output = model(input_ids)
        torch.cuda.synchronize()
        by_tag, by_name = timer.summary()
    """

    def __init__(self) -> None:
        self._handles: list = []
        self._records: List[Tuple[str, str, torch.cuda.Event, torch.cuda.Event]] = []
        self._stacks: Dict[int, list] = defaultdict(list)
        self.enabled: bool = False

    # ------------------------------------------------------------------
    # Module-level hooks (inclusive time for the whole module)
    # ------------------------------------------------------------------
    def add_module(self, module: torch.nn.Module, tag: str, name: str) -> None:
        """Register pre/post forward hooks on *module*.

        The recorded interval is the wall-clock CUDA time from the moment the
        module starts executing on GPU until it returns.
        """

        def _pre_hook(_mod, _inputs):
            if not self.enabled:
                return
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._stacks[id(_mod)].append(start)

        def _post_hook(_mod, _inputs, _output):
            if not self.enabled:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = self._stacks[id(_mod)].pop()
            self._records.append((tag, name, start, end))

        self._handles.append(module.register_forward_pre_hook(_pre_hook))
        self._handles.append(module.register_forward_hook(_post_hook))

    # ------------------------------------------------------------------
    # Manual range API (for non-module regions, e.g., attention core)
    # ------------------------------------------------------------------
    @contextmanager
    def range(self, tag: str, name: str):
        """Context manager for instrumenting a code region that is NOT an nn.Module.

        Example::

            with timer.range("attn_core", "L0.attn_core"):
                output = flash_attn_func(q, k, v)
        """
        if not self.enabled:
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._records.append((tag, name, start, end))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Discard all recorded events (but keep hooks registered)."""
        self._records.clear()
        self._stacks.clear()

    def remove(self) -> None:
        """Remove all registered hooks and release references."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._records.clear()
        self._stacks.clear()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """Aggregate elapsed time (ms) by tag and by (tag, name).

        Returns
        -------
        by_tag : dict  {tag: total_ms}
        by_name : dict {(tag, name): total_ms}
        """
        by_tag: Dict[str, float] = defaultdict(float)
        by_name: Dict[Tuple[str, str], float] = defaultdict(float)

        for tag, name, start, end in self._records:
            ms = start.elapsed_time(end)
            by_tag[tag] += ms
            by_name[(tag, name)] += ms

        return dict(by_tag), dict(by_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def cleanup_cuda() -> None:
        """Aggressively free CUDA memory.  Call between profiling runs."""
        gc.collect()
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
