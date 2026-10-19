"""Shared test isolation.

Recall hook markers default to ``tempfile.gettempdir()/dream-recall-<uid>``,
the same directory the real hooks use. ``dream ingest`` now harvests (and
deletes) old markers there, so every test gets its own marker directory and
can never count or remove a real session's markers. Tests that set
``DREAM_RECALL_STATE_ROOT`` themselves still win.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_recall_marker_root(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path_factory.mktemp("recall-markers")))
