import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stage_runner
import metrics
import sys


class StageRunnerTest(unittest.TestCase):
    def test_continued_failure_runs_later_stage_and_remains_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            metrics.start_run(state_path, now=100.0)
            with patch.object(stage_runner, "STATE_PATH", state_path):
                result = stage_runner.run_stage(
                    "ingest", [sys.executable, "-c", "raise SystemExit(1)"],
                    continue_on_error=True,
                )
                self.assertEqual(result, 0)
                self.assertEqual(metrics._read_state(state_path)["status"], "running")
                self.assertEqual(stage_runner.run_stage(
                    "distill", [sys.executable, "-c", "pass"], continue_on_error=True,
                ), 0)
            state = metrics.finish_run(state_path, success=True)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stage"], "ingest")
            self.assertEqual(state["completed_stages"], ["distill"])
            self.assertEqual(state["failed_stages"], [{"stage": "ingest", "error_code": "stage_exit_1"}])
            self.assertEqual(state["started_at"], 100.0)

    def test_continue_flag_does_not_suppress_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(stage_runner, "STATE_PATH", Path(tmp) / "state.json"):
                def interrupt(_command):
                    raise KeyboardInterrupt()
                with self.assertRaises(KeyboardInterrupt):
                    stage_runner.run_stage("ingest", ["fake"], runner=interrupt,
                                           continue_on_error=True)

    def test_successful_command_marks_stage_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            with patch.object(stage_runner, "STATE_PATH", state_path):
                result = stage_runner.run_stage(
                    "ingest",
                    ["fake", "ingest"],
                    runner=lambda _command: 0,
                )

            self.assertEqual(result, 0)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["stage"], "ingest")
            self.assertEqual(state["completed_stages"], ["ingest"])

    def test_failed_command_marks_safe_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            with patch.object(stage_runner, "STATE_PATH", state_path):
                result = stage_runner.run_stage(
                    "distill",
                    ["fake", "distill"],
                    runner=lambda _command: 17,
                )

            self.assertEqual(result, 17)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stage"], "distill")
            self.assertEqual(state["error_code"], "stage_exit_17")


if __name__ == "__main__":
    unittest.main()
