# Codex LLM Suggestion Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Dream suggestion resolution into the nightly Codex pipeline using `gpt-5.6-luna` with high reasoning, while preserving backups, SHA conflict safety, and host-side write controls.

**Architecture:** Add a provider-neutral `review` stage routed locally to the existing `codex` provider. A focused curation module builds a bounded prompt and validates strict structured decisions (`accept`, `reject`, `merge`, `defer`); `dream.py` remains the only owner of database/file writes and reuses the guarded apply helpers. The systemd unit invokes the new curation command, while the old deterministic apply command remains available only for explicit manual use.

**Tech Stack:** Python 3, `tomllib`, existing provider-neutral `backend.generate`, Codex CLI provider, SQLite, pytest, systemd user service.

**Spec:** `docs/superpowers/specs/2026-08-21-codex-llm-suggestion-curation-design.md`

## Global Constraints

- Production review route is `provider = "codex"`, `model = "gpt-5.6-luna"`, `reasoning_effort = "high"`.
- The nightly path must not invoke Claude or deterministic accept-all.
- The model cannot choose target paths or bypass host-side path, protected-file, backup, or SHA checks.
- A provider, timeout, or schema error must produce no writes from that curation run.
- `MEMORY.md` keeps append-only index merge semantics; regular memory files use complete replacement bodies.
- Existing explicit `accept`, `reject`, `merge`, and deterministic `apply-configured` CLI contracts remain compatible.

### Task 1: Add the Codex review stage and curation decision contract

**Files:**
- Modify: `dream/default-config.toml`
- Modify: `dream/config.py` only if validation/default handling requires it
- Create: `dream/curate.py`
- Test: `dream/test_provider_architecture.py`, `dream/test_curate.py`

**Interfaces:**
- Consumes: existing `DreamConfig`, `backend.generate`, and pending suggestion snapshots.
- Produces: `CURATION_SCHEMA`, `build_curation_prompt(rows, memory_root)`, and `parse_curation_output(output, suggestion_ids)` for the CLI task.

- [ ] **Step 1: Write failing configuration and decision-contract tests.**

```python
def test_default_review_stage_routes_to_codex_luna_high():
    config = load_config(Path("/nonexistent/dream-config.toml"))
    review = config.stage("review")
    assert review["provider"] == "codex"
    assert review["model"] == "gpt-5.6-luna"
    assert review["reasoning_effort"] == "high"


def test_parse_curation_output_requires_one_valid_decision_per_id():
    from curate import parse_curation_output

    result = parse_curation_output(
        {"decisions": [
            {"suggestion_id": 7, "decision": "merge", "body": "merged", "reason": "keeps newer facts"},
        ]},
        {7},
    )
    assert result[7].decision == "merge"
    assert result[7].body == "merged"


def test_parse_curation_output_rejects_missing_or_unknown_ids():
    from curate import parse_curation_output
    with pytest.raises(ValueError, match="suggestion IDs"):
        parse_curation_output(
            {"decisions": [{"suggestion_id": 8, "decision": "defer", "reason": "unclear"}]},
            {7},
        )
```

- [ ] **Step 2: Run the focused tests and verify they fail for missing stage/contract behavior.**

Run: `python -m pytest -q -p no:cacheprovider dream/test_provider_architecture.py dream/test_curate.py`

Expected: FAIL because the `review` stage and `curate` module do not exist yet.

- [ ] **Step 3: Add the review stage and strict decision module.**

Add this default configuration:

```toml
[stages.review]
provider = "codex"
model = "gpt-5.6-luna"
reasoning_effort = "high"
timeout_seconds = 1800
```

Implement `curate.py` with a strict schema whose only decision values are
`accept`, `reject`, `merge`, and `defer`; require `suggestion_id`, `decision`,
and `reason`, allow `body` only as a string, reject duplicate/missing/unknown
IDs, and build prompts that include the target kind/path, proposal, rationale,
source sessions, current body, current SHA, stored base SHA, and conflict flag.
The prompt must state that the model must never invent paths and that index
responses contain only intended index lines.

- [ ] **Step 4: Run the focused tests and the provider import guard.**

Run: `python -m pytest -q -p no:cacheprovider dream/test_provider_architecture.py dream/test_curate.py`

Expected: PASS, including proof that loading the review route creates the Codex provider without importing Claude.

- [ ] **Step 5: Commit Task 1.**

```bash
git add dream/default-config.toml dream/config.py dream/curate.py dream/test_provider_architecture.py dream/test_curate.py
git commit -m "feat: add Codex review stage contract"
```

### Task 2: Implement the LLM curation command with guarded writes

**Files:**
- Modify: `dream/dream.py` in suggestion command dispatch and helpers
- Test: `dream/test_curate.py`, `dream/tests/test_review_helpers.py`, `dream/test_cli_failures.py`

**Interfaces:**
- Consumes: `CURATION_SCHEMA`, `build_curation_prompt`, `parse_curation_output`, existing `_apply_suggestion`, `_backup_memory`, `_resolved_body`, and `_prune_orphaned_index_lines`.
- Produces: `dream suggestions curate-configured [--dry-run]`, with model decisions applied only after validation and SHA recheck.

- [ ] **Step 1: Write failing command tests using a fake structured Codex response.**

Cover these exact behaviors:

```python
def test_curate_accept_merge_reject_and_defer_updates_only_expected_rows(tmp_path, monkeypatch):
    memory_root, conn = make_review_fixture(tmp_path)
    monkeypatch.setattr("dream.backend.generate", fake_curation_response)
    assert run_curate_configured(memory_root, conn) == 0
    assert read_target(memory_root, "accept.md") == "proposed accept body"
    assert read_target(memory_root, "merge.md") == "model merged body"
    assert read_target(memory_root, "reject.md") == "original reject body"
    assert statuses(conn) == {1: "accepted", 2: "accepted", 3: "rejected", 4: "pending"}


def test_curate_conflict_rechecks_sha_and_defers_without_overwrite(tmp_path, monkeypatch):
    memory_root, conn = make_review_fixture(tmp_path)
    monkeypatch.setattr("dream.backend.generate", response_after_target_changes)
    assert run_curate_configured(memory_root, conn) == 0
    assert read_target(memory_root, "conflict.md") == "new live body"
    assert statuses(conn)[5] == "pending"


def test_curate_provider_failure_writes_nothing(tmp_path, monkeypatch):
    memory_root, conn = make_review_fixture(tmp_path)
    before = snapshot_fixture(memory_root, conn)
    monkeypatch.setattr("dream.backend.generate", raise_provider_error)
    assert run_curate_configured(memory_root, conn) != 0
    assert snapshot_fixture(memory_root, conn) == before
    assert not (memory_root / "backups").exists()
```

Use a temporary memory root and SQLite fixture; monkeypatch only the provider
boundary, not filesystem behavior that the test is meant to exercise.

- [ ] **Step 2: Run the focused command tests and verify the expected failures.**

Run: `python -m pytest -q -p no:cacheprovider dream/test_curate.py dream/tests/test_review_helpers.py`

Expected: FAIL because `curate-configured` is not registered and no curation orchestration exists.

- [ ] **Step 3: Implement the curation orchestration.**

Add a `curate-configured` branch to `cmd_suggestions` that:

1. Loads all pending rows and snapshots current file bodies, SHA values, and conflict flags.
2. Calls `backend.generate("review", prompt, CURATION_SCHEMA, config=args.config)` once for the batch.
3. Parses and validates a complete one-decision-per-ID response before changing any row.
4. Supports `--dry-run` with decision output and zero writes.
5. Creates one memory backup lazily before the first actual file write.
6. Rechecks each target SHA immediately before applying an `accept` or `merge`; on change, prints a deferral and leaves the row pending.
7. For a valid model `merge`, refreshes the suggestion base SHA/body under the current target snapshot, then invokes `_apply_suggestion` so path containment, protected-file, backup, preview cleanup, and recall synchronization remain host-controlled.
8. Marks `reject` in the DB without touching the target; leaves `defer` pending.
9. Returns nonzero for provider/schema failures, but reports per-suggestion deferrals without bypassing safety.

The existing `apply-configured` branch remains unchanged and deterministic for explicit emergency use.

- [ ] **Step 4: Run focused and full tests.**

Run: `python -m pytest -q -p no:cacheprovider dream/test_curate.py dream/tests/test_review_helpers.py dream/test_cli_failures.py`

Expected: PASS.

Then run: `python -m pytest -q -p no:cacheprovider dream`

Expected: all existing tests plus the new curation tests pass.

- [ ] **Step 5: Commit Task 2.**

```bash
git add dream/dream.py dream/curate.py dream/test_curate.py dream/tests/test_review_helpers.py dream/test_cli_failures.py
git commit -m "feat: curate Dream suggestions through Codex"
```

### Task 3: Switch the deployed nightly path and document the behavior

**Files:**
- Modify: `dream/systemd/dream.service`
- Modify: `README.md`
- Modify: `skill/SKILL.md`
- Modify: `docs/superpowers/specs/2026-08-21-codex-llm-suggestion-curation-design.md` only if implementation details require clarification
- Test: `dream/test_provider_architecture.py` or a new service/config assertion test

**Interfaces:**
- Consumes: `dream suggestions curate-configured` and the review stage configuration.
- Produces: a deployed unit that invokes Codex curation and never Claude/deterministic apply in the nightly chain.

- [ ] **Step 1: Write a failing service/config assertion.**

```python
def test_nightly_service_invokes_codex_curation_not_mechanical_apply():
    service = (ROOT / "systemd" / "dream.service").read_text(encoding="utf-8")
    assert "dream suggestions curate-configured" in service
    assert "dream suggestions apply-configured" not in service
    assert "claude" not in service.lower()
```

- [ ] **Step 2: Run it and verify it fails against the current unit.**

Run: `python -m pytest -q -p no:cacheprovider dream/test_provider_architecture.py`

Expected: FAIL because the current unit still invokes `apply-configured`.

- [ ] **Step 3: Switch the unit and update operator documentation.**

Replace the final unit command with:

```ini
ExecStart=%h/.local/bin/dream suggestions curate-configured
```

Document the Codex review stage, model, decision types, dry-run command,
conflict deferral, and the fact that `apply-configured` is deterministic and no
longer part of the nightly path. Keep existing safety and rollback guidance.

- [ ] **Step 4: Run service/config tests and whitespace checks.**

Run: `python -m pytest -q -p no:cacheprovider dream/test_provider_architecture.py`

Expected: PASS.

Run: `git diff --check HEAD~2..HEAD`

Expected: no output.

- [ ] **Step 5: Commit Task 3.**

```bash
git add dream/systemd/dream.service README.md skill/SKILL.md dream/test_provider_architecture.py
git commit -m "feat: use Codex curation in nightly Dream"
```

### Task 4: Deploy local configuration, exercise Codex, and resolve current pending rows

**Files:**
- Modify outside Git: `/home/szymon/.config/dream/config.toml`
- Verify outside Git: `/home/szymon/.config/systemd/user/dream.service`
- Test runtime state: `/home/szymon/.claude/projects/-home-szymon/memory/.suggestions/` and `~/.claude/dream.db`

**Interfaces:**
- Consumes: the committed curation command and local review stage.
- Produces: current local config routed to Codex Luna high, a successful dry-run, a real LLM curation run for existing pending suggestions, and a reloaded timer.

- [ ] **Step 1: Back up and update local Dream config without touching unrelated settings.**

Create a timestamped mode-preserving backup outside Git, then add or update only:

```toml
[stages.review]
provider = "codex"
model = "gpt-5.6-luna"
reasoning_effort = "high"
timeout_seconds = 1800
```

Verify the current file is otherwise equal to its backup.

- [ ] **Step 2: Run a Codex curation dry-run.**

Run:

```bash
dream suggestions curate-configured --dry-run
```

Expected: a structured Codex-backed decision report for the current pending rows; no memory file, DB status, preview, or backup changes.

- [ ] **Step 3: Run the real curation and inspect the result.**

Run:

```bash
dream suggestions curate-configured
```

Expected: output identifies the Codex provider/model, decisions, applied/rejected/deferred counts, and backup path. Verify every applied file is inside the memory root, no current SHA conflict was bypassed, and deferred rows remain pending.

- [ ] **Step 4: Reload the user timer and verify its command chain.**

Run:

```bash
cp -p dream/systemd/dream.service /home/szymon/.config/systemd/user/dream.service
systemctl --user daemon-reload
systemctl --user cat dream.service
```

Expected: the active unit contains `dream suggestions curate-configured` and no Claude or deterministic apply command. If the restricted shell cannot talk to the user bus, record that limitation and verify the deployed file directly.

- [ ] **Step 5: Commit no personal config; record runtime evidence.**

Keep the local config backup outside Git, run `dream status`, `dream preflight`, and the full test suite, and record provider/model, pending/accepted counts, backup path, and any deferred conflicts in the implementation report.

### Task 5: Final verification and review

**Files:**
- Verify all changed repository files and runtime config; do not modify Dream DB manually.

- [ ] **Step 1: Run the complete verification set.**

```bash
python -m pytest -q -p no:cacheprovider dream
git diff --check 24d3553..HEAD
dream preflight
dream status
```

- [ ] **Step 2: Verify source and deployed command separation.**

```bash
rg -n 'curate-configured|apply-configured|claude|gpt-5.6-luna' dream README.md skill /home/szymon/.config/dream/config.toml /home/szymon/.config/systemd/user/dream.service
```

The nightly chain must contain the Codex curation command, the review stage must
show `gpt-5.6-luna` and `high`, and the unit must not contain a Claude command.

- [ ] **Step 3: Run final hostile review and report the known acceptance evidence.**

Review the full change range and the runtime evidence. Confirm no test relies
on a real provider for deterministic assertions, no user-local config is
committed, and all unresolved/deferred suggestions are explicitly reported.

- [ ] **Step 4: Commit any review fixes, rerun the full suite, and leave the repository clean.**
