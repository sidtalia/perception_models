"""Background prompt embedding: atomic swap, coalescing, errors."""

from __future__ import annotations

import threading
import time

import pytest

from perception_models.prompt_bank import PromptBank


def _wait(pred, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_initial_blocking_and_snapshot():
    bank = PromptBank(lambda ps: [len(p) for p in ps])
    try:
        snap = bank.set_blocking(["mug", "banana"])
        assert snap.version == 1 and snap.prompts == ("mug", "banana") and snap.cond == [3, 6]
        assert bank.current() is snap
    finally:
        bank.close()


def test_update_runs_off_caller_thread_and_old_snapshot_stays_until_ready():
    gate = threading.Event()
    seen_threads = []

    def embed(ps):
        seen_threads.append(threading.current_thread().name)
        if ps == ("slow",):
            gate.wait(5)
        return list(ps)

    bank = PromptBank(embed)
    try:
        first = bank.set_blocking(["fast"])
        assert bank.request(["slow"]) is True
        time.sleep(0.05)
        assert bank.current() is first  # hot path keeps using the old prompts
        gate.set()
        assert _wait(lambda: bank.current().prompts == ("slow",))
        assert all(n == "prompt_bank" for n in seen_threads)
    finally:
        bank.close()


def test_requests_coalesce_to_newest():
    gate = threading.Event()
    calls = []

    def embed(ps):
        calls.append(ps)
        if ps == ("a",):
            gate.wait(5)
        return ps

    bank = PromptBank(embed)
    try:
        bank.request(["a"])
        assert _wait(lambda: calls == [("a",)])
        bank.request(["b"])
        bank.request(["c"])
        gate.set()
        assert _wait(lambda: bank.current() is not None and bank.current().prompts == ("c",))
        assert ("b",) not in calls
    finally:
        bank.close()


def test_rejects_empty_unchanged_and_over_capacity():
    bank = PromptBank(lambda ps: ps, max_prompts=2)
    try:
        assert bank.request(["", "  "]) is False
        bank.set_blocking(["x"])
        assert bank.request(["x"]) is False
        with pytest.raises(ValueError):
            bank.request(["a", "b", "c"])
    finally:
        bank.close()


def test_embedding_error_keeps_previous_snapshot():
    errors = []

    def embed(ps):
        if ps == ("bad",):
            raise RuntimeError("boom")
        return ps

    bank = PromptBank(embed, on_error=lambda ps, e: errors.append((ps, str(e))))
    try:
        good = bank.set_blocking(["good"])
        bank.request(["bad"])
        assert _wait(lambda: errors)
        assert bank.current() is good
        assert bank.request(["after"])  # worker still alive
        assert _wait(lambda: bank.current().prompts == ("after",))
    finally:
        bank.close()
