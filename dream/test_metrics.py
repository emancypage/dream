import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import metrics


class MetricsStateTest(unittest.TestCase):
    def test_multiple_continued_failures_survive_later_success_and_finalize(self):
        state_path = self._path()
        metrics.start_run(state_path, now=100.0)
        for stage, stamp in [("ingest", 101.0), ("distill", 102.0)]:
            metrics.start_stage(state_path, stage=stage, now=stamp)
            state = metrics.finish_stage(
                state_path, stage=stage, success=False, now=stamp,
                error_code="provider_failed", continue_on_error=True,
            )
            self.assertNotIn("finished_at", state)
        metrics.start_stage(state_path, stage="curation", now=103.0)
        metrics.finish_stage(state_path, stage="curation", success=True, now=104.0)
        metrics.finish_run(state_path, success=True, now=110.0)
        state = metrics.finalize_run(state_path, service_result="success", exit_status="0", now=111.0)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stage"], "ingest")
        self.assertEqual(state["started_at"], 100.0)
        self.assertEqual(state["finished_at"], 110.0)
        self.assertEqual(state["completed_stages"], ["curation"])
        self.assertEqual([failure["stage"] for failure in state["failed_stages"]], ["ingest", "distill"])
        self.assertIn("dream_run_health 0", metrics.render_metrics(state, now=111.0))

    def test_start_run_writes_running_state(self):
        state_path = self._path()

        state = metrics.start_run(state_path, now=100.0)

        self.assertEqual(state["status"], "running")
        self.assertEqual(state["started_at"], 100.0)
        self.assertEqual(state["stage"], "starting")
        self.assertEqual(json.loads(state_path.read_text(encoding="utf-8")), state)

    def test_stage_failure_preserves_stage_and_safe_error_code(self):
        state_path = self._path()
        metrics.start_run(state_path, now=100.0)

        state = metrics.finish_stage(
            state_path,
            stage="distill",
            success=False,
            now=105.0,
            error_code="provider_failed",
        )

        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stage"], "distill")
        self.assertEqual(state["finished_at"], 105.0)
        self.assertEqual(state["error_code"], "provider_failed")
        self.assertNotIn("prompt", json.dumps(state).lower())

    def test_success_records_completed_stage_and_duration(self):
        state_path = self._path()
        metrics.start_run(state_path, now=100.0)
        metrics.finish_stage(state_path, stage="ingest", success=True, now=102.0)

        state = metrics.finish_run(state_path, success=True, now=110.0)

        self.assertEqual(state["status"], "success")
        self.assertEqual(state["stage"], "curation")
        self.assertEqual(state["finished_at"], 110.0)
        self.assertEqual(state["duration_seconds"], 10.0)

    def test_seed_success_imports_verified_systemd_timestamps(self):
        state_path = self._path()

        state = metrics.seed_success(
            state_path,
            started_at=100.0,
            finished_at=110.0,
            source="systemd",
        )

        self.assertEqual(state["status"], "success")
        self.assertEqual(state["source"], "systemd")
        self.assertEqual(state["started_at"], 100.0)
        self.assertEqual(state["finished_at"], 110.0)
        self.assertEqual(state["completed_stages"], list(metrics.STAGES))

    def test_finalize_marks_unexpected_service_failure(self):
        state_path = self._path()
        metrics.start_run(state_path, now=100.0)

        state = metrics.finalize_run(
            state_path,
            service_result="signal",
            exit_status="TERM",
            now=106.0,
        )

        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error_code"], "service_signal_TERM")
        self.assertEqual(state["finished_at"], 106.0)

    def test_finalize_does_not_hide_stage_failure(self):
        state_path = self._path()
        metrics.start_run(state_path, now=100.0)
        metrics.finish_stage(
            state_path,
            stage="distill",
            success=False,
            now=105.0,
            error_code="stage_exit_17",
        )

        state = metrics.finalize_run(
            state_path,
            service_result="exit-code",
            exit_status="1",
            now=106.0,
        )

        self.assertEqual(state["stage"], "distill")
        self.assertEqual(state["error_code"], "stage_exit_17")

    def test_health_maps_state_to_dashboard_value(self):
        cases = [
            ({"status": "success", "finished_at": 100.0}, 101.0, 3),
            ({"status": "running", "started_at": 100.0}, 101.0, 2),
            ({"status": "success", "finished_at": 100.0}, 100.0 + 36 * 3600 + 1, 1),
            ({"status": "failed", "finished_at": 100.0}, 101.0, 0),
            ({}, 101.0, -1),
        ]
        for state, now, expected in cases:
            with self.subTest(state=state):
                self.assertEqual(
                    metrics.health_value(state, now=now, stale_after=36 * 3600),
                    expected,
                )

    def test_render_metrics_exposes_one_hot_status_and_safe_info_labels(self):
        state = {
            "status": "failed",
            "stage": "distill",
            "started_at": 100.0,
            "finished_at": 110.0,
            "duration_seconds": 10.0,
            "error_code": "provider_failed",
        }

        rendered = metrics.render_metrics(state, now=120.0, stale_after=36 * 3600)

        self.assertIn("dream_run_health 0", rendered)
        self.assertIn('dream_run_status{status="failed"} 1', rendered)
        self.assertIn(
            'dream_last_run_info{result="failed",stage="distill",error_code="provider_failed"} 1',
            rendered,
        )
        self.assertIn("dream_last_run_timestamp_seconds 110", rendered)
        self.assertIn("dream_last_run_duration_seconds 10", rendered)

    def test_render_metrics_reports_stale_without_exposing_error_text(self):
        state = {
            "status": "success",
            "stage": "curation",
            "finished_at": 100.0,
            "error": "secret transcript content",
        }

        rendered = metrics.render_metrics(state, now=100.0 + 36 * 3600 + 1, stale_after=36 * 3600)

        self.assertIn("dream_run_health 1", rendered)
        self.assertIn('dream_run_status{status="stale"} 1', rendered)
        self.assertNotIn("secret transcript content", rendered)

    def test_state_write_is_atomic_enough_to_leave_valid_json(self):
        state_path = self._path("nested", "dream-status.json")

        metrics.start_run(state_path, now=time.time())

        self.assertTrue(state_path.exists())
        json.loads(state_path.read_text(encoding="utf-8"))

    def _path(self, *parts):
        root = Path(tempfile.mkdtemp(prefix="dream-metrics-test-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        return root.joinpath(*parts) if parts else root / "dream-status.json"


if __name__ == "__main__":
    unittest.main()
