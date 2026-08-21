# Codex Memories Single-Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Dream the only automatic local memory-recall layer for Codex on this machine, while disabling native Codex Memories safely and preserving a complete rollback path.

**Architecture:** Dream remains canonical: `~/.claude/dream.db` stores the recall index and `~/.claude/projects/-home-szymon/memory` stores approved memory files. Codex native Memories are explicitly disabled for both reading existing memories and generating new ones; Dream preflight reads the Codex configuration so an empty scaffold directory is not mistaken for active duplicate injection. Existing Dream hooks remain the only automatic context injection path.

**Tech Stack:** Python 3, `tomllib`, SQLite/FTS5, pytest, Codex CLI `config.toml`, Codex hooks JSON.

**Spec:** `docs/superpowers/specs/2026-08-20-automatic-memory-recall-design.md`

## Global Constraints

- Keep `~/.claude/dream.db` and the configured Dream memory root as the canonical persistent stores.
- Set both native Codex controls explicitly: `memories.use_memories = false` and `memories.generate_memories = false`.
- Do not edit, merge, or delete generated files under `~/.codex/memories`; the current directory contains no memory payload files, but the cleanup must remain reversible.
- Do not change the two existing Dream hook commands or add launch flags, writable-directory flags, or a background Dream service.
- Hook recall remains fail-open and read-only against the persistent SQLite database.
- Review hook trust through Codex `/hooks`; do not silently trust arbitrary commands.
- All external configuration changes require a timestamped backup before the first write.

---

### Task 1: Replace path-existence detection with an explicit native-memory state check

**Files:**
- Modify: `dream/recall_context.py:70-104`
- Test: `dream/test_recall_preflight.py`

**Interfaces:**
- Consumes: `Path.home()`, Codex `config.toml`, and the existing `recall_preflight(config, db_path)` call.
- Produces: `codex_memories_check(codex_home: Path | None = None) -> tuple[bool, str]`, used by `recall_preflight` to produce the existing `codex.memories-double-injection` check.

- [ ] **Step 1: Write the failing tests for explicit native-memory states**

Add tests to `dream/test_recall_preflight.py` using a temporary fake Codex home:

```python
def test_empty_codex_memories_scaffold_is_not_duplicate_injection(tmp_path):
    from recall_context import codex_memories_check

    codex_home = tmp_path / "codex"
    (codex_home / "memories" / ".agents").mkdir(parents=True)
    (codex_home / "memories" / ".codex").mkdir()

    ok, detail = codex_memories_check(codex_home)

    assert ok
    assert "disabled" in detail or "empty" in detail


def test_enabled_codex_memories_is_reported_as_duplicate_injection(tmp_path):
    from recall_context import codex_memories_check

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "[features]\nmemories = true\n[memories]\nuse_memories = true\n",
        encoding="utf-8",
    )
    (codex_home / "memories").mkdir()

    ok, detail = codex_memories_check(codex_home)

    assert not ok
    assert "enabled" in detail


def test_explicitly_disabled_codex_memories_passes_even_with_old_files(tmp_path):
    from recall_context import codex_memories_check

    codex_home = tmp_path / "codex"
    memories = codex_home / "memories"
    memories.mkdir(parents=True)
    (memories / "old.md").write_text("preserved generated state", encoding="utf-8")
    (codex_home / "config.toml").write_text(
        "[features]\nmemories = false\n[memories]\n"
        "use_memories = false\ngenerate_memories = false\n",
        encoding="utf-8",
    )

    ok, detail = codex_memories_check(codex_home)

    assert ok
    assert "disabled" in detail
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
python -m pytest -q -p no:cacheprovider dream/test_recall_preflight.py
```

Expected: FAIL because `codex_memories_check` does not exist and the current preflight only checks whether the directory exists.

- [ ] **Step 3: Implement the smallest read-only state checker**

In `dream/recall_context.py`, import `tomllib` and add `codex_memories_check` with these rules:

1. Resolve `codex_home` to `Path.home() / ".codex"` when omitted.
2. Parse `config.toml` only if it exists; malformed or unreadable configuration must return `(False, "Codex Memories configuration could not be read: ...")`.
3. Treat native Memories as enabled only when the feature flag is `true` and neither relevant setting explicitly disables it. The explicit settings are `[memories] use_memories` and `[memories] generate_memories`.
4. If both controls are false, return success with a detail that says native Codex Memories are disabled. Do not inspect generated memory content beyond determining that the directory is not an active source.
5. If the controls are not enabled but the directory exists only with `.agents/`, `.codex/`, or no regular payload files, return success with a detail that identifies an empty/stale scaffold.
6. If native Memories can read or generate memories, return failure with a detail that names the active control and directs the operator to disable native Memories before relying on Dream.

Replace the current direct `Path.home() / ".codex" / "memories"` existence check with:

```python
memories_ok, memories_detail = codex_memories_check()
checks.append(("codex.memories-double-injection", memories_ok, memories_detail))
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```bash
python -m pytest -q -p no:cacheprovider dream/test_recall_preflight.py
```

Expected: PASS, including the existing read-only preflight tests.

- [ ] **Step 5: Commit the code and tests**

```bash
git add dream/recall_context.py dream/test_recall_preflight.py
git commit -m "fix: detect active Codex Memories safely"
```

### Task 2: Disable native Codex Memories with a reversible configuration change

**Files:**
- Modify: `$CODEX_HOME/config.toml` or `$HOME/.codex/config.toml` (outside the repository; never commit this file)
- Backup: `$CODEX_HOME/backups/codex-memories-disable-<timestamp>/config.toml`
- Preserve: `$CODEX_HOME/memories/` without editing its generated contents

**Interfaces:**
- Consumes: Codex’s documented `[features] memories` flag and `[memories]` controls.
- Produces: a local Codex configuration in which native Memories neither inject existing memories nor generate new memory inputs.

- [ ] **Step 1: Capture a timestamped backup and inventory the current generated state**

Run from a normal host shell, not from a restricted hook sandbox:

```bash
codex_home="$HOME/.codex"
configured_codex_home="$(printenv CODEX_HOME 2>/dev/null || true)"
if test -n "$configured_codex_home"; then codex_home="$configured_codex_home"; fi
stamp="$(date +%Y%m%d-%H%M%S)"
backup_dir="$codex_home/backups/codex-memories-disable-$stamp"
install -d -m 700 "$backup_dir"
cp -p "$codex_home/config.toml" "$backup_dir/config.toml"
find "$codex_home/memories" -maxdepth 3 -type f -print 2>/dev/null | sort > "$backup_dir/memory-files.txt"
du -sh "$codex_home/memories" > "$backup_dir/memory-size.txt" 2>/dev/null || true
```

Expected for this machine: `memory-files.txt` is empty; only the existing empty scaffold directories are present. If regular memory files are found, stop before moving or deleting anything and review them through Codex `/memories`.

- [ ] **Step 2: Add explicit native-memory-off settings**

Preserve all unrelated keys in `config.toml` and add/update exactly:

```toml
[features]
memories = false

[memories]
use_memories = false
generate_memories = false
```

If `[features]` already exists, add only `memories = false`; do not duplicate the table. Keep the existing `js_repl` setting. If `[memories]` already exists, change only the two controls above and preserve unrelated memory settings.

- [ ] **Step 3: Start a fresh Codex session and confirm the native layer is off**

In a new interactive Codex session, run `/memories` and confirm that existing local memories are not used and the current chat is not eligible to generate new memories. The config is the global control; the `/memories` choice is the per-chat verification.

- [ ] **Step 4: Run Dream preflight against the real store**

Run:

```bash
dream preflight
```

Expected: `codex.memories-double-injection` is `ok` with a detail indicating native Memories are disabled. The empty `~/.codex/memories` scaffold remains untouched and no Dream data moves.

### Task 3: Document the single-source policy and rollback

**Files:**
- Modify: `README.md`
- Modify: `skill/SKILL.md`
- Modify: `docs/superpowers/specs/2026-08-20-automatic-memory-recall-design.md`

**Interfaces:**
- Consumes: the configuration and preflight behavior from Tasks 1–2.
- Produces: operator documentation that explains which store owns which responsibility and how to restore native Memories deliberately.

- [ ] **Step 1: Document ownership and boundaries**

Add concise sections stating:

- Dream owns persistent approved memories, distilled summaries, lexical indexing, and automatic hook recall.
- Codex native Memories are disabled locally with `memories.use_memories = false` and `memories.generate_memories = false`.
- `~/.codex/memories` is generated Codex state and is not indexed, edited, or merged by Dream.
- `dream preflight` reports native Memories as a failure when `[features].memories = true` and either native operation is not explicitly disabled, but does not fail merely because an empty scaffold directory exists.
- Required behavior belongs in `AGENTS.md` or checked-in documentation, not in generated memory state.

- [ ] **Step 2: Document rollback without destructive commands**

Document the exact rollback sequence using the backup directory created in Task 2:

```bash
codex_home="$HOME/.codex"
configured_codex_home="$(printenv CODEX_HOME 2>/dev/null || true)"
if test -n "$configured_codex_home"; then codex_home="$configured_codex_home"; fi
cp -p "$codex_home/backups/codex-memories-disable-<timestamp>/config.toml" "$codex_home/config.toml"
```

State that restoring native Memories is a deliberate choice: run `/memories` in a new session, verify the desired `use_memories` and `generate_memories` behavior, then do not use Dream’s automatic recall simultaneously unless duplicate injection is intentionally accepted.

- [ ] **Step 3: Commit the repository documentation**

```bash
git add README.md skill/SKILL.md docs/superpowers/specs/2026-08-20-automatic-memory-recall-design.md
git commit -m "docs: define Dream as the single memory source"
```

### Task 4: Verify hook behavior and the no-duplication contract

**Files:**
- Test: existing Dream test suite and hook smoke commands
- Verify: `/home/szymon/.codex/hooks.json` without committing personal configuration

**Interfaces:**
- Consumes: the existing `UserPromptSubmit` and `SessionStart` Dream hooks plus the read-only recall path.
- Produces: evidence that new sessions use Dream automatically, without launch flags or native-memory duplication.

- [ ] **Step 1: Verify the hook file still contains exactly the two Dream handlers**

Run:

```bash
jq '[.hooks.UserPromptSubmit[], .hooks.SessionStart[] | .hooks[]? | select(.command | startswith("dream context"))] | length' /home/szymon/.codex/hooks.json
```

Expected: `2`. Review the commands in Codex `/hooks` and trust only the exact Dream handlers if Codex requests trust again.

Also verify the pre-existing style hook was preserved and still emits the short-response rules:

```bash
jq -e '.hooks.UserPromptSubmit[0].hooks[0].command == "bash '\''/home/szymon/.codex/hooks/style-reminder.sh'\''"' /home/szymon/.codex/hooks.json
bash /home/szymon/.codex/hooks/style-reminder.sh | jq -e '.hookSpecificOutput.hookEventName == "UserPromptSubmit" and (.hookSpecificOutput.additionalContext | contains("Maksymalnie 5 zdań"))'
```

Expected: both commands succeed; the first `UserPromptSubmit` group remains the style hook and its output still contains the five-sentence rule.

- [ ] **Step 2: Run the complete automated suite**

Run:

```bash
python -m pytest -q -p no:cacheprovider dream
git diff --check HEAD~2..HEAD
```

Expected: all tests pass and no whitespace errors are reported.

- [ ] **Step 3: Run live recall smoke tests with unique session IDs**

Run:

```bash
printf '%s\n' '{"session_id":"single-source-session-start","source":"startup","cwd":"/home/szymon/Dev/skills/dream","repository_roots":["/home/szymon/Dev/skills/dream"]}' | dream context session-start
printf '%s\n' '{"session_id":"single-source-prompt","prompt":"What Dream memory rules apply here?","cwd":"/home/szymon/Dev/skills/dream","repository_roots":["/home/szymon/Dev/skills/dream"]}' | dream context prompt
```

Expected: each command exits zero, emits at most one JSON hook response, and the rendered context carries Dream’s untrusted-reference marker. The persistent database and its backup files remain unchanged during both hook calls.

- [ ] **Step 4: Record the final acceptance state**

Acceptance requires all of the following:

- `dream preflight` has no failed native-memory check.
- A fresh Codex session shows native Memories disabled through `/memories`.
- `~/.codex/memories` has not been deleted or edited; its inventory is preserved in the timestamped backup.
- `hooks.json` still has the two Dream handlers and no extra launch flags.
- The pre-existing `style-reminder.sh` remains the first `UserPromptSubmit` hook and still emits the concise-response rules.
- Full tests pass and the live hooks remain fail-open/read-only.
- `git status --short` is clean in the Dream repository; user-local Codex config and backups remain outside Git.

## Rollback

If the Codex configuration change causes an unexpected behavior, restore the timestamped `config.toml` backup, start a fresh Codex session, and rerun `dream preflight`. Do not remove `~/.codex/memories`; its generated contents are preserved and can be inspected or re-enabled later through Codex’s documented controls.
