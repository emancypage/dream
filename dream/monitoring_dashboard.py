#!/usr/bin/env python3
"""Idempotently add Dream health panels to the RPi Fleet dashboard."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence


PROMETHEUS = {"type": "prometheus", "uid": "prometheus"}
OWNED_PANEL_IDS = frozenset({40, 41, 42, 43})
HEALTH_QUERY = 'dream_run_health{job="dream",instance="dream"}'
AGE_QUERY = 'time() - dream_last_run_timestamp_seconds{job="dream",instance="dream"}'
INFO_QUERY = 'dream_last_run_info{job="dream",instance="dream"} == 1'


def _target(expression: str, *, instant: bool, legend: str = "") -> dict[str, Any]:
    return {
        "datasource": PROMETHEUS,
        "editorMode": "code",
        "expr": expression,
        "format": "time_series",
        "instant": instant,
        "legendFormat": legend,
        "range": not instant,
        "refId": "A",
    }


def _thresholds() -> dict[str, Any]:
    return {
        "mode": "absolute",
        "steps": [
            {"color": "red", "value": None},
            {"color": "orange", "value": 1},
            {"color": "blue", "value": 2},
            {"color": "green", "value": 3},
        ],
    }


def _health_mappings() -> list[dict[str, Any]]:
    return [
        {
            "type": "value",
            "options": {
                "-1": {"text": "BRAK DANYCH", "color": "gray"},
                "0": {"text": "ERROR", "color": "red"},
                "1": {"text": "NIEAKTUALNY", "color": "orange"},
                "2": {"text": "W TRAKCIE", "color": "blue"},
                "3": {"text": "OK", "color": "green"},
            },
        }
    ]


def _stat(
    panel_id: int,
    title: str,
    grid: dict[str, int],
    target: dict[str, Any],
    *,
    unit: str = "short",
    mappings: list[dict[str, Any]] | None = None,
    thresholds: dict[str, Any] | None = None,
    text_mode: str = "auto",
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "color": {"mode": "thresholds"},
        "mappings": mappings or [],
        "thresholds": thresholds or {
            "mode": "absolute",
            "steps": [{"color": "green", "value": None}],
        },
        "unit": unit,
    }
    return {
        "id": panel_id,
        "type": "stat",
        "title": title,
        "gridPos": grid,
        "datasource": PROMETHEUS,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "colorMode": "value",
            "graphMode": "none",
            "justifyMode": "auto",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "showPercentChange": False,
            "textMode": text_mode,
            "wideLayout": True,
        },
        "targets": [target],
    }


def _panels(y: int) -> list[dict[str, Any]]:
    health = _stat(
        40,
        "Dream — Stan nocnego przebiegu",
        {"x": 0, "y": y, "w": 12, "h": 5},
        _target(HEALTH_QUERY, instant=True),
        mappings=_health_mappings(),
        thresholds=_thresholds(),
    )
    age = _stat(
        41,
        "Dream — Wiek ostatniego przebiegu",
        {"x": 12, "y": y, "w": 12, "h": 5},
        _target(AGE_QUERY, instant=True, legend="wiek"),
        unit="s",
    )
    age["fieldConfig"]["defaults"]["thresholds"] = {
        "mode": "absolute",
        "steps": [{"color": "green", "value": None}, {"color": "orange", "value": 129600}],
    }
    stage = _stat(
        42,
        "Dream — Etap i wynik",
        {"x": 0, "y": y + 5, "w": 24, "h": 5},
        _target(INFO_QUERY, instant=True, legend="{{result}} · {{stage}}"),
        text_mode="name",
    )
    history = {
        "id": 43,
        "type": "timeseries",
        "title": "Dream — Historia zdrowia",
        "gridPos": {"x": 0, "y": y + 10, "w": 24, "h": 8},
        "datasource": PROMETHEUS,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "decimals": 0,
                "max": 3,
                "min": -1,
                "thresholds": _thresholds(),
                "unit": "short",
            },
            "overrides": [],
        },
        "options": {
            "legend": {
                "calcs": ["lastNotNull"],
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"hideZeros": False, "mode": "single", "sort": "none"},
        },
        "targets": [_target(HEALTH_QUERY, instant=False, legend="health")],
    }
    return [health, age, stage, history]


def _next_y(panels: list[dict[str, Any]]) -> int:
    bottom = 0
    for panel in panels:
        grid = panel.get("gridPos")
        if not isinstance(grid, dict):
            continue
        try:
            bottom = max(bottom, int(grid.get("y", 0)) + int(grid.get("h", 0)))
        except (TypeError, ValueError):
            continue
    return bottom


def patch_dashboard(source: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(source)
    panels = result.get("panels")
    if not isinstance(panels, list):
        raise ValueError("dashboard panels must be a list")
    retained = [panel for panel in panels if panel.get("id") not in OWNED_PANEL_IDS]
    result["panels"] = retained + _panels(_next_y(retained))
    version = result.get("version", 0)
    result["version"] = (version if isinstance(version, int) else 0) + 1
    return result


def atomic_write(path: Path, content: str) -> None:
    temporary_name = ""
    mode = path.stat().st_mode
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    source = json.loads(args.path.read_text(encoding="utf-8"))
    patched = patch_dashboard(source)
    atomic_write(args.path, json.dumps(patched, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
