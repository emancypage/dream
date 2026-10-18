# Character pipeline implementation plan

> For agentic workers: use superpowers:subagent-driven-development or superpowers:executing-plans. Follow the test-first steps below. Szymon explicitly authorized immediate execution after this plan is committed.

## Goal

Implement M1–M8 with human-owned character rules, generated preference pairs,
independent judging, explicit human acceptance, context deployment, evaluation,
training export, and lineage-aware retention.

## Architecture

Use `dream/character/` for character operations and separate SQLite character
tables. Reuse configuration, provider requests, read-only storage, and retention.
Host functions own identities, paths, backups, transitions, and transactions.

## Tech stack

Python 3.11+ standard library, SQLite, argparse, TOML, Markdown, JSON, pytest.

## Spec

`docs/superpowers/specs/2026-10-18-character-pipeline-design.md`, approved revision.

## Global constraints

- Work on `feat/character`. Preserve the untracked `session-logs/` files.
- No local configuration edits, real model calls, push, or PR.
- Runtime uses Python 3.11+ standard library only. Tests use fake providers.
- Human approval is mandatory for constitution and every deployed pair.
- Provider commands never appear in systemd. Provider routing belongs in config.
- Character defaults to disabled; hook injection defaults to false.
- Both new nightly limits are 10. Render defaults are 3 pairs and 12000 codepoints.
- Seeds are optional for approval. Preserve the three Polish examples exactly.
- Never write the constitution after initial draft installation, retention policy,
  AGENTS.md, CLAUDE.md, or native Codex memory state.
- Read commands use `storage.open_db_readonly`, without migration or WAL fallback.
- Retention filters before calls, before writes, and at every artifact read.
- Run `pytest -q -p no:cacheprovider dream` and `git diff --check` before every commit.

## Review focus

- A symlink or hardlink output must not overwrite a protected input. Task 1 tests
  resolved destination identities and inode aliases.
- A policy or constitution edit during a provider call must discard that item's
  result. Tasks 2 and 3 test stale preconditions before persistence.
- A false correction near the start of the queue must not starve later work.
  Task 2 tests saved negative dispositions and pre-limit oversized skips.
- A new judgment during retention backup must abort the whole character cleanup.
  Task 6 tests changed affected sets, with no partial character deletion.
- A probe with a reused trait ID but changed meaning must not get a score.
  Task 5 tests exact trait-text binding and unsupported reporting.

## Shared interfaces

`character.constitution.load_constitution(config, require_approved=True)` returns
an immutable document with text, sha256, version, traits, seeds, and approved_at.
`character.safety.policy_for(config)` loads the memory-root policy.
`character.safety.check_constitution(document, policy)` refuses excluded text.
`character.safety.guard_path(path, config, *, output_root=None)` rejects protected
or escaping identities. `write_artifacts(config, files, manifest)` backs up old
files and replaces host-selected files atomically.

`character.storage.records(conn, table)` and `get_record(conn, table, id)` return
plain dictionaries. `insert_record` and `update_record` preserve exact revision,
hash, constitution, and provenance fields. Every mutation uses a snapshot and
backups. Tables store validated record payloads as canonical JSON with indexed
host identity, work key, revision, and content hash. Structured payloads contain
the fields named by the spec. This keeps nested message/provenance objects exact.
`eligible_records(conn, config, table, constitution=None)` applies policy, lineage,
and optional constitution matching. `snapshot(conn)` supports retention comparisons.
`open_writer(config)` checks database paths before using `dream.open_db` and backs
up before existing character mutations. Character schemas are created by v3.

Provider code uses `backend.generate(stage, prompt, schema, config=config,
system_prompt=...)`; `validate_output` plus semantic checks validate fakes too.
All character calls append policy context. No provider output controls paths or
approval state. Public functions return JSON-compatible reports with `failures`.
Database errors propagate; item provider errors collect and continue.

### Task 1. M1 and shared host contracts

Files: `character/__init__.py`, `constitution.py`, `types.py`, `safety.py`,
`storage.py`, `constitution.template.md`, `config.py`, `default-config.toml`,
`schema.sql`, `dream.py` migration, `backend.py`, `install.sh`.
Tests: `test_character_constitution.py`, `test_character_config.py`,
`test_character_safety.py`, `test_character_storage.py`.

- [ ] Write tests for draft refusal, seedless approval, exact template text,
  unknown/type-invalid config, family routing, aliases, read-only no-writes,
  migration backup before DDL, and unchanged existing review defaults.
- [ ] Run focused tests and confirm missing implementation fails.
- [ ] Implement the shared interfaces above. Add the strict character section and
  four provider-only routes. Add optional backend system_prompt and stage-scoped
  preflight. Implement v3 pre-DDL backup and additive tables.
- [ ] Validate all required sections, stable IDs, timestamp, and optional seeds.
  Installer only creates an absent draft with no symlink overwrite.
- [ ] Run focused tests, full suite, and whitespace checks. Commit
  `feat: add character constitution and host contracts`.

### Task 2. M2 generation and cost estimate

Files: `character/generate.py`, `prompts/character-extract-v1.md`,
`character-expand-v1.md`, `character-response-v1.md`.
Tests: `test_character_generate.py`.
Interfaces: `estimate(conn, config, *, source, limit, refresh=False,
expand_seeds=False) -> dict`; `generate_pairs(conn, config, *, source, limit,
refresh=False, expand_seeds=False) -> dict`.

- [ ] Write failing tests for English/Polish cues, visible bounded context,
  exclusions before calls, seedless skips, negative dispositions, limits,
  no correction leakage, synthetic arm isolation, provenance, refresh attempts,
  provider/schema failures, input races, and fatal storage errors.
- [ ] Confirm failures, then implement shared selection for estimate and generate.
  Provider responses require host schema/identity/length checks. Keep conflicts.
  Both synthetic arms commit together. No failed success cursor is saved.
- [ ] Include source='claude' historical-family evidence. Reject equal responses.
- [ ] Run focused and full tests, whitespace check, then commit
  `feat: generate character preference pairs`.

### Task 3. M3 judging and M4 manual review

Files: `character/judge.py`, `character/review.py`, `prompts/character-judge-v1.md`.
Tests: `test_character_judge.py`, `test_character_review.py`.
Interfaces: `judge_pairs(conn, config, *, limit) -> dict`;
`list_pairs(conn, config) -> list`; `show_pair(conn, config, id) -> dict`;
`review_pair(conn, config, id, action, *, expected_sha, reason='',
body=None, acknowledge_flags=False) -> dict`.

- [ ] Test two swapped calls, normalized verdicts/assessments, failed second calls,
  mandatory family checks versus extract/generate, recorded override, and history
  overlap warning. Confirm tests fail first.
- [ ] Implement advisory judgments tied to pair revision/hash. No auto-accept.
- [ ] Test and implement revision-bound human transitions. Accept requires current
  judgment and explicit flag acknowledgement. Edits clear judgment and acceptance,
  require a reason, and preserve original rejected evidence. Stale reads do not write.
- [ ] Run focused/full tests and whitespace check. Commit
  `feat: judge and review character pairs`.

### Task 4. M5 rendering, M7 export, and session-start hook

Files: `character/render.py`, `export.py`, `dream.py`, `recall_documents.py`,
`consolidate.py`; hook/context tests and character render/export/security tests.
Interfaces: `render_block(conn, config) -> dict` with text/manifest;
`export_pairs(conn, config) -> dict` with text/manifest; previews do not write.

- [ ] Write failing tests for deterministic bytes, full constitution budget,
  trait coverage and ties, accepted-current-revision eligibility, policy changes,
  protected outputs, exact DPO fields, and source overlap provenance.
- [ ] Implement complete-example selection and stable dream-dpo-v1 serialization.
  Back up and atomically write fixed filenames under validated state root.
- [ ] Test and implement separate trusted session-start composition. Reuse the
  pure renderer, respect hook total limit, retain relevance-ranked untrusted recall,
  five-second timeout, compact behavior, fail-open handling, and default-off flag.
- [ ] Exclude protected constitution identities and aliases from consolidation
  input, recall sync, and read-time recall. Never promote recalled text to character.
- [ ] Run focused/full tests and whitespace check. Commit
  `feat: render and export approved character context`.

### Task 5. M6 paired fixed-probe evaluation

Files: `character/targets.py`, `eval.py`, `fixtures/character/public.json`,
`prompts/character-eval-v1.md`, `character-eval-judge-v1.md`.
Tests: `test_character_eval.py`.
Interfaces: target protocol `identity()` and `respond(messages, context_block,
request_settings)`; `evaluate(conn, config, *, fixtures, limit) -> dict`;
`report_run(conn, config, run_id) -> dict` with JSON data and Markdown table.

- [ ] Write failing tests for multi-turn trajectories, same baseline/A route,
  trait-text hashes, unsupported skips, no public-fixture generation, controls,
  strict 0..4 scoring, swapped disagreement, partial failure, common scored subset,
  retained call provenance, and current-policy report filtering.
- [ ] Implement baseline and A through the existing provider API. Add only the
  target interface for future B. Store generated responses separately from valid
  completed scores. Failed second judgment saves no scored result.
- [ ] Keep reports private and deterministic apart from recorded run metadata.
- [ ] Run focused/full tests and whitespace check. Commit
  `feat: evaluate character context against fixed probes`.

### Task 6. M8 retention, CLI, nightly integration, and documentation

Files: `character/retention.py`, `cli.py`, `retention_cleanup.py`, `dream.py`,
`metrics.py`, `systemd/dream.service`, monitoring tests, README, skill docs.
Tests: `test_character_retention.py`, `test_character_cli.py`,
`test_nightly_failure_isolation.py`, `test_provider_architecture.py`.

- [ ] Test preview affected lineage across every state, backup contents, deletion,
  all-or-nothing race rollback, draft/missing constitution, protected managed files,
  user-modified outputs, and unchanged source transcripts. Confirm failures first.
- [ ] Integrate character snapshot into the same affected-records.json backup.
  Recompute affected set/hashes inside write transaction. A mismatch deletes no
  character rows and reports rerun_required. Keep old memory cleanup semantics.
- [ ] Wire all character CLI commands with read-only versus writable openers,
  exact required flags, JSON output, safe exit codes, and clean disabled skips.
- [ ] Test and add separate character_generate and character_judge nightly stages.
  Extend fixed metrics tuple and last-stage completion logic. Verify no literal
  provider commands or human review/deployment commands run in systemd.
- [ ] Document config, approval, seedless behavior, provider data flows, limits,
  manual workflow, hook budgets, retention, copied-file limits, and DPO format.
- [ ] Run focused/full tests and whitespace check. Commit
  `feat: integrate character retention and nightly commands`.

### Task 7. Final branch review

- [ ] Review the complete diff against every M1–M8 requirement. Use a fresh reviewer.
- [ ] Fix material findings with tests and separate logical commits.
- [ ] Run full hermetic suite, CLI help and preview checks, and git diff --check.
- [ ] Report commit IDs, test count, and any unfinished result. Do not push or
  change local installed configuration. Do not run a real model smoke test.
