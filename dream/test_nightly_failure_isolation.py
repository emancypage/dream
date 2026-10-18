import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import metrics


ROOT = Path(__file__).parent


@pytest.mark.parametrize("failed_stage", metrics.STAGES)
def test_scheduled_commands_attempt_every_stage_and_report_failure(tmp_path, failed_stage):
    command = tmp_path / "fixture_cli.py"
    command.write_text("""
import os, sys
from pathlib import Path
stage = ('character_' + sys.argv[2]) if sys.argv[1] == 'character' else ('curation' if sys.argv[1] == 'suggestions' else sys.argv[1])
with Path(os.environ['DREAM_TEST_RECORD']).open('a') as record:
    record.write(stage + '\\n')
raise SystemExit(1 if stage == os.environ['DREAM_TEST_FAILURE'] else 0)
""")
    state_path = tmp_path / "run.json"
    record_path = tmp_path / "attempted.txt"
    env = dict(os.environ, DREAM_METRICS_STATE=str(state_path),
               DREAM_TEST_RECORD=str(record_path), DREAM_TEST_FAILURE=failed_stage)
    metrics.start_run(state_path)
    service = (ROOT / "systemd" / "dream.service").read_text()
    for line in service.splitlines():
        if not line.startswith("ExecStart="):
            continue
        args = shlex.split(line.split("=", 1)[1])
        assert args[0] == "%h/.local/bin/dream-stage"
        cli_index = args.index("%h/.local/bin/dream")
        args = [sys.executable, str(ROOT / "stage_runner.py"), *args[1:cli_index],
                sys.executable, str(command), *args[cli_index + 1:]]
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    metrics.finish_run(state_path, success=True)
    state = metrics.finalize_run(state_path, service_result="success", exit_status="0")
    assert record_path.read_text().splitlines() == list(metrics.STAGES)
    assert state["status"] == "failed"
    assert state["stage"] == failed_stage
    assert state["failed_stages"] == [{"stage": failed_stage, "error_code": "stage_exit_1"}]
    assert state["completed_stages"] == [stage for stage in metrics.STAGES if stage != failed_stage]
    assert json.loads(state_path.read_text()) == state
