# dream — provider-agnostic local memory consolidation

`dream` ingests agent transcripts, filters them to visible user/assistant turns, distils sessions into structured findings, and consolidates those findings into reviewable persistent-memory suggestions.

Transcript sources and model providers are independent. The packaged profile reads Codex JSONL and Open Code SQLite sessions and runs both model stages through `codex exec`; Claude JSONL and Claude CLI remain optional adapters.

## Requirements

- **Python 3.11 or newer** — the runtime uses only the standard library (`tomllib` requires 3.11+).
- **Codex CLI** — the default provider for both model stages. Install it, log in with your ChatGPT subscription (`codex login`), and keep it on `PATH`. `dream preflight` verifies the executable and the ChatGPT auth file without calling the model.
- **`~/.local/bin` on `PATH`** — the installer symlinks the CLI to `~/.local/bin/dream`.
- **Optional: Claude CLI** — only needed if you configure a stage with the `claude-cli` provider type. A missing `claude` executable does not affect Codex-backed stages, ingest, status, or search.
- **Platform: Linux or macOS.** The bundled systemd units for nightly runs are Linux-only.
- **pytest** is a development dependency only (used by the test suite and CI); it is not needed to run `dream`.

## Install

```bash
git clone https://github.com/emancypage/dream.git
cd dream
./install.sh
```

The installer creates two symlinks into your **current checkout directory** and, on first install, copies the default configuration:

- `dream/dream.py` → `~/.local/bin/dream`
- `skill/` → `~/.codex/skills/dream`
- `dream/default-config.toml` is copied to `~/.config/dream/config.toml`, only when no configuration file exists yet.

Because the links point at the checkout, update in place:

```bash
cd dream   # wherever you cloned it
git pull
```

Updating via `git pull` is safe: the installer never writes into the repository, and an existing `~/.config/dream/config.toml` is left untouched on every install.

## Configuration

The default path is `~/.config/dream/config.toml`. `install.sh` creates it from `dream/default-config.toml` only when it does not already exist.

```toml
[stages.distill]
provider = "codex"
reasoning_effort = "low"

[stages.consolidate]
provider = "codex"
reasoning_effort = "high"

[providers.codex]
type = "codex-cli"
auth = "chatgpt-subscription"

[[sources]]
name = "codex"
type = "codex-jsonl"
root = "~/.codex/sessions"
enabled = true

[[sources]]
name = "opencode"
type = "opencode-sqlite"
root = "~/.local/share/opencode"
enabled = true

[review]
mode = "suggest-only"
```

`model` is **optional** in each stage: when omitted, the provider's default model is used. Set it explicitly when you want a specific model:

```toml
[stages.distill]
provider = "codex"
model = "my-model-id"
```

Model names belong to configuration, not the core. Configuration precedence is CLI override (`dream distill --model X` / `dream consolidate --model X`), stage-specific environment variable (`DREAM_DISTILL_MODEL`, `DREAM_CONSOLIDATE_MODEL`), TOML, then packaged defaults. Unknown provider/source types and unknown options fail explicitly.

Changing provider/model configuration does not automatically re-run every historical session. Use `dream distill --refresh` for an explicit provenance refresh; changed transcript revisions are always re-distilled.

Legacy compatibility variables remain accepted: `DREAM_BACKEND`, `CLAUDE_DREAM_DB`, and `DREAM_MEMORY_ROOT`.

## Privacy and data flow

- **Local reads.** `dream ingest` and `dream search` read only local transcript stores (by default Codex JSONL under `~/.codex/sessions` and Open Code SQLite at `~/.local/share/opencode/opencode.db`) and the local Dream SQLite database. They never transmit anything.
- **Model calls send content.** The `distill` and `consolidate` stages send selected session content to the model provider configured for that stage — by default the Codex CLI against your ChatGPT subscription. Distillation sends the filtered transcript of a session; consolidation sends the distilled notes plus the current memory store. The review stage is routed through the configured Codex provider; the production profile is `gpt-5.6-luna` with `reasoning_effort = "high"` and sends pending suggestions plus bounded target context.
- **Everything else stays local.** The SQLite database (`~/.claude/dream.db` by default), suggestion previews (`.suggestions/`), and the memory store remain on your machine.
- **Check your configuration first.** Before the first run, review the `[[sources]]` roots and the `[stages.*]`/`[providers.*]` entries in `~/.config/dream/config.toml` so you know which transcripts are read and where their content is sent.
- **Local and external effects.** `estimate`, `search`, `status`, `preflight`, `context`, and `suggestions list` only read the persistent local data. Automatic recall opens SQLite through an immutable read-only connection; its small deduplication markers live under `/tmp`. `ingest` reads transcripts and updates the local SQLite index without calling a model. `distill`, `consolidate`, and configured curation make model calls and write local results. Suggestion review commands update local review state; `accept`, `merge`, and configured apply operations can also update memory files, with backups before live-memory writes.
- **Cost / subscription usage.** Each model call can incur cost or consume part of your subscription limit. `dream estimate -v` shows how many sessions a distillation run would cover before you run it.
- **Consent.** Do not run `dream` on confidential transcripts unless you have consciously agreed to send their selected content to the configured provider.

## Architecture

```text
Codex skill / shell / systemd
                     |
                neutral CLI
                     |
         +-----------+-----------+
         |           |           |
transcript sources  stage router  SQLite + memory
         |           |           |
codex-jsonl        model provider  suggestions
opencode-sqlite    codex/claude    conflict-safe apply
claude-jsonl
```

The pipeline stages are:

1. `ingest`: discover configured sources (Codex JSONL, Open Code SQLite, optional Claude JSONL), filter to visible user/assistant turns, and index messages with SQLite FTS5.
2. `distill`: call the provider configured for the `distill` stage once per changed session revision.
3. `consolidate`: call the configured `consolidate` provider and produce suggestions.
4. `suggestions`: inspect and resolve suggestions with file-version preconditions and backups.

Consolidation uses an explicit ledger of consumed distillation identities. It does not use session timestamps as its progress cursor and does not silently consume truncated batches.

Ingestion isolates source initialization, discovery, and session parsing errors.
Healthy sources and sessions still enter the index. Failed items keep their old
revision, if any, so a later run can retry them. The CLI logs each source error and
exits with status 1 after processing the other sources. Dream database write
errors remain fatal to ingestion.

The nightly service runs each stage through `dream-stage --continue-on-error`.
If a source or model provider fails, later stages can process previously stored
transcripts, successful distillations, and pending suggestions. Failed model
calls do not create validated results or advance their consolidation ledger.
The service completes its remaining commands, but run metrics retain every
failed stage and report failed health rather than a clean success. Direct CLI
commands and `dream-stage` without that flag retain their nonzero exit status.
Interruptions and terminated stage processes still stop the service.

The OpenCode adapter supports the legacy workspace `branch` column and the
new workspace schema that omits it. Missing branch metadata becomes `null`;
transcript text continues to enter the index.

## Layout

- `dream/` — the Python runtime: CLI, tests, configuration defaults, prompts, and systemd units.
- `skill/` — the Codex skill: `SKILL.md` and `agents/openai.yaml`.

## Testing

pytest is a development dependency. Run the suite from the repository root:

```bash
pip install -r requirements-dev.txt
pytest -q -p no:cacheprovider dream
```

The suite is hermetic: it never invokes the Codex or Claude CLIs, requires no credentials, and makes no model calls.

## Usage

```bash
dream ingest
dream estimate -v
dream distill --yes
dream consolidate
dream status
dream preflight
dream context session-start < hook-payload.json
dream context prompt < hook-payload.json
dream hooks install
dream hooks uninstall
dream recall-eval --fixtures dream/fixtures/recall/public.json
```

### Excluding a topic from local memory

Create `.dream-retention.json` inside the configured memory root. This hidden,
user-controlled file is excluded from automatic recall and cannot be changed by
generated memory suggestions. For example:

```json
{
  "instructions": "Do not retain the retired project, including paraphrases from old sessions.",
  "memory_paths": ["project_retired.md"],
  "project_slugs": ["retired-project"],
  "text_markers": ["unique-retired-project-marker"]
}
```

Dream passes the instructions to distillation, consolidation, and curation.
Project exclusions prevent distillation. Exact file and project matches, plus
case-insensitive text markers, also filter generated facts, proposals, writes,
index synchronization, and recall. The prose instruction handles semantic scope;
the deterministic checks do not recognize every possible paraphrase.

Run `dream retention` to preview removal of existing generated memories, then
`dream retention --apply` to apply it. The command backs up affected records
outside the memory root, removes listed topic files and their index entries,
clears excluded generated facts, rejects matching pending proposals, and refreshes
the recall index. Source transcripts and existing chat messages remain unchanged.
Pending proposals also retain their source-project exclusions. Cleanup defers
files and distilled records changed after its snapshot and reports them for retry.
Policies are local to the configured memory root and do not affect another machine.

## Memory ownership and Codex Memories

Dream is the canonical single automatic memory-recall layer. It owns persistent
approved memories, distilled summaries, their lexical index, and automatic
hook recall. Native Codex Memories are disabled locally with these controls:

```toml
[memories]
use_memories = false
generate_memories = false
```

`~/.codex/memories` is generated Codex state. Dream does not index, edit, merge,
or delete that directory or its contents. An empty scaffold directory is not
active native memory state, so it does not make `dream preflight` fail;
Preflight fails only when `[features].memories = true` and either
`memories.use_memories` or `memories.generate_memories` is not explicitly
`false`. Required behavior belongs in `AGENTS.md` or checked-in
documentation, never in generated memory state.

### Deliberately restoring native Memories

If native Memories must be restored, use the timestamped backup made before
the configuration change. This sequence copies the backup back without
deleting anything:

```bash
codex_home="$HOME/.codex"
configured_codex_home="$(printenv CODEX_HOME 2>/dev/null || true)"
if test -n "$configured_codex_home"; then codex_home="$configured_codex_home"; fi
cp -p "$codex_home/backups/codex-memories-disable-<timestamp>/config.toml" "$codex_home/config.toml"
```

Then run `/memories` in a new session and verify the desired
`use_memories` and `generate_memories` behavior. Do not use Dream automatic
recall simultaneously unless duplicate injection is intentionally accepted.

Automatic recall is disabled at the Codex hook boundary until `dream hooks install`
has been reviewed and trusted through Codex `/hooks`. The hook is read-only with
respect to `~/.claude/dream.db`, so it works in Codex's workspace sandbox without
extra launch flags or writable-home configuration. The host-side systemd timer
remains the write path for ingest, distillation, consolidation, and memory review.
The lexical store is local, raw transcripts are disabled by default, and rendered
results are marked as untrusted reference data with source provenance. `recall.diagnostic_path` receives
bounded JSONL diagnostics; diagnostics never contain raw document bodies.

### Codex hook timeout

`dream hooks install` generates both Dream hooks with a five-second timeout. The
timeout applies to `SessionStart` and `UserPromptSubmit`; it accommodates the
local recall command, which can take a few seconds while starting Python and
reading the SQLite index. The generated setting is written to the user-local
`~/.codex/hooks.json`, not to this repository; `dream/recall_hooks.py` is the
repository source of truth. Re-run `dream hooks install` after upgrading an
installation that was created with an older timeout.

Optional `recall.embedder` and `recall.reranker` adapters are disabled by default,
cache by content hash and adapter fingerprint, accept at most 100 candidates, and
fall back to lexical results on unavailable adapters, invalid output, missing
calibration, or adapter errors. Set `remote_data_egress = true` only after
reviewing the adapter's data-flow implications; Dream does not add a model package
or network dependency.

Search indexed conversations:

```bash
dream search "postgres migration"
dream search "docker daemon ip" --role assistant
dream search 'NEAR("dream" "memory", 5)' --project -home-alice-Dev-api
```

Machine-readable review:

```bash
dream suggestions list
dream suggestions curate-configured --dry-run
dream suggestions curate-configured
dream suggestions accept 12
dream suggestions reject 13
dream suggestions merge 14 --body-file /tmp/merged.md
dream suggestions apply-configured
```

For autonomous user-requested review, use `curate-configured` and report each model decision and reason. The `--dry-run` form is the no-write preview; the form without it is the automatic route used by the nightly service. The model chooses `accept`, `reject`, `merge`, or `defer`, while host-side checks retain authority over target paths, SHA preconditions, protected files, append-only `MEMORY.md` index updates, and backups. Provider or schema failure aborts before writes, and a concurrent target change is deferred for a later run. `accept` and `merge` still refuse to write when the target changed after suggestion creation.

`apply-configured` remains an explicit deterministic emergency/fallback command, is not the nightly path, and requires `review.mode = "auto-apply"`; in `suggest-only` mode it exits successfully without writing. The explicit `accept`, `reject`, and `merge` commands remain available for targeted manual decisions.

The legacy `dream review` interactive diff remains available.

## Data and compatibility

Existing paths remain the defaults during migration:

- DB: `~/.claude/dream.db`;
- memory: `~/.claude/projects/<home-slug>/memory/` (the packaged default resolves the home slug for the running user).

These are storage locations, not runtime dependencies on Claude Code. Schema upgrades are additive, versioned, and create a consistent SQLite backup before modifying a legacy on-disk database.

Each session records source, external identity, content revision, and parser version. Each distillation records source revision, prompt version, provider, model, usage when available, and duration.

## Scheduling (Linux)

`dream/systemd/dream.service` runs:

```text
dream ingest
dream distill --yes --limit 50
dream consolidate
dream suggestions curate-configured
```

The review stage asks the configured Codex provider for accept/reject/merge/defer decisions. Provider routing and review mode come from TOML; the unit contains no provider/model selection. Production uses `gpt-5.6-luna` with `reasoning_effort = "high"`.

```bash
./install.sh
systemctl --user daemon-reload
systemctl --user enable --now dream.timer
```

On macOS, schedule the same commands with `launchd` or cron.

## Monitoring (Linux)

The scheduled service records a bounded JSON state file at
`~/.local/state/dream/run.json`. Each stage is wrapped by `dream-stage`; a
failure preserves the failing stage and a safe error code. The separate
`dream-exporter.service` exposes only numeric Prometheus metrics and never
exports provider error text or transcript contents.

The exporter should bind to an address reachable by Prometheus but not to every
network interface. On this setup it listens on the laptop's Tailscale address
`100.85.174.35:9409`. The main metric is `dream_run_health`: `3` = success,
`2` = running, `1` = stale, `0` = failed, and `-1` = no recorded run.

The test-backed patchers add the exporter target and four panels to the existing
fleet dashboard:

```bash
python3 dream/monitoring_prometheus.py /path/to/prometheus.yml 100.85.174.35:9409
python3 dream/monitoring_dashboard.py /path/to/rpi-fleet.json
```

Both patchers are idempotent, reject ambiguous targets, and preserve unrelated
dashboard panels. Back up the live files before applying generated output.

## Main files

Paths are relative to `dream/`.

| Path | Purpose |
|---|---|
| `config.py`, `default-config.toml` | strict configuration and deployment defaults |
| `model_types.py`, `backend.py` | neutral generation contract and stage routing |
| `providers/` | lazily loaded model-provider adapters |
| `sources/` | transcript-source adapters |
| `ingest.py` | neutral ingestion and SQLite writes |
| `distill.py` | per-session structured extraction |
| `consolidate.py` | ledger-backed consolidation and suggestion persistence |
| `dream.py` | CLI, migrations, search, backups and review |
| `schema.sql` | current SQLite schema |
| `prompts/` | provider-neutral prompts |
| `recall_documents.py`, `recall_query.py` | canonical synchronization and lexical retrieval |
| `recall_context.py`, `recall_hooks.py` | fail-open context execution and Codex hook management |
| `recall_eval.py`, `recall_adapters.py` | fixture evaluation, calibration, and optional adapter contracts |
| `metrics.py`, `stage_runner.py` | run-state persistence and Prometheus exporter |
| `monitoring_prometheus.py`, `monitoring_dashboard.py` | idempotent Prometheus and Grafana patchers |
