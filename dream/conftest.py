"""Shared test isolation and latency budgets.

Recall hook markers default to ``tempfile.gettempdir()/dream-recall-<uid>``,
the same directory the real hooks use. ``dream ingest`` now harvests (and
deletes) old markers there, so every test gets its own marker directory and
can never count or remove a real session's markers. Tests that set
``DREAM_RECALL_STATE_ROOT`` themselves still win.

Latency tests check their p95 with ``assert_latency(name, p95_ms, budget_ms)``.
The budget at scale 1 is the acceptance gate; ``DREAM_PERF_BUDGET_SCALE``
(default 3 when ``CI`` is set, otherwise 1) multiplies it only so that slow
shared runners still catch regressions. Every check prints one
``perf: <name> p95=<ms> budget=<ms> scale=<x>`` line (shown with ``pytest -s``).
"""

import os

import pytest


def perf_budget_scale() -> float:
    """``DREAM_PERF_BUDGET_SCALE``, or 3 on CI (``CI`` set) and 1 elsewhere."""
    return float(os.environ.get("DREAM_PERF_BUDGET_SCALE", "3" if os.environ.get("CI") else "1"))


@pytest.fixture
def assert_latency():
    """``assert_latency(name, p95_ms, budget_ms)``: print the raw numbers, then check the scaled budget."""
    scale = perf_budget_scale()

    def check(name: str, p95_ms: float, budget_ms: float) -> None:
        effective = budget_ms * scale
        print(f"perf: {name} p95={p95_ms:.1f} budget={budget_ms:g} scale={scale:g}")
        assert p95_ms <= effective, (f"{name}: p95 {p95_ms:.1f} ms is over the effective budget {effective:g} ms "
                                     f"(budget {budget_ms:g} ms x DREAM_PERF_BUDGET_SCALE {scale:g})")

    return check


@pytest.fixture(autouse=True)
def _isolated_recall_marker_root(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path_factory.mktemp("recall-markers")))
