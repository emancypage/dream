import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stage_runner


class StageRunnerTest(unittest.TestCase):
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
