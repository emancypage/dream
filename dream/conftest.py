"""Shared test isolation and latency budgets.

Recall hook markers default to ``tempfile.gettempdir()/dream-recall-<uid>``,
the same directory the real hooks use. ``dream ingest`` now harvests (and
deletes) old markers there, so every test gets its own marker directory and
can never count or remove a real session's markers. Tests that set
``DREAM_RECALL_STATE_ROOT`` themselves still win.

Latency tests state their budgets for a developer machine and compare against
``perf_budget_ms(budget)``, which multiplies by ``DREAM_PERF_BUDGET_SCALE``
(default 3 when ``CI`` is set, otherwise 1), because shared CI runners are
slower.
"""

import os

import pytest


def perf_budget_scale() -> float:
    """``DREAM_PERF_BUDGET_SCALE``, or 3 on CI (``CI`` set) and 1 elsewhere."""
    return float(os.environ.get("DREAM_PERF_BUDGET_SCALE", "3" if os.environ.get("CI") else "1"))


@pytest.fixture
def perf_budget_ms():
    """``perf_budget_ms(milliseconds)``: a latency budget scaled for the machine running the suite."""
    scale = perf_budget_scale()
    return lambda milliseconds: milliseconds * scale


@pytest.fixture(autouse=True)
def _isolated_recall_marker_root(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path_factory.mktemp("recall-markers")))
