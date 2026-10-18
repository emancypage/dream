from pathlib import Path
import metrics


def test_nightly_character_stages_are_separate_provider_neutral_commands(tmp_path):
    assert metrics.STAGES == ('ingest', 'distill', 'consolidate', 'curation', 'character_generate', 'character_judge')
    service = (Path(__file__).parent / 'systemd/dream.service').read_text()
    for stage, command in [('character_generate', 'generate --source all --limit 10'), ('character_judge', 'judge --limit 10')]:
        assert f'dream-stage --continue-on-error {stage} -- %h/.local/bin/dream character {command}' in service
    for action in ('accept', 'edit', 'review', 'render', 'export', 'eval'):
        assert f'dream character {action}' not in service
    for provider in ('codex exec', 'claude '):
        assert provider not in service
    path = tmp_path / 'run.json'
    metrics.start_run(path)
    assert metrics.finish_run(path, success=True)['stage'] == 'character_judge'
    assert metrics.seed_success(path, started_at=1, finished_at=2)['stage'] == 'character_judge'
