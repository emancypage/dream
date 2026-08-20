# dream — provider-agnostic local memory consolidation

`dream` ingests agent transcripts, filters them to visible user/assistant turns, distils sessions into structured findings, and consolidates those findings into reviewable persistent-memory suggestions.

Transcript sources and model providers are independent. The packaged profile reads Codex JSONL and runs both model stages through `codex exec`; Claude JSONL and Claude CLI remain optional adapters.

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

- **Local reads.** `dream ingest` and `dream search` read only local transcript files (by default `~/.codex/sessions`) and the local SQLite database. They never transmit anything.
- **Model calls send content.** The `distill` and `consolidate` stages send selected session content to the model provider configured for that stage — by default the Codex CLI against your ChatGPT subscription. Distillation sends the filtered transcript of a session; consolidation sends the distilled notes plus the current memory store.
- **Everything else stays local.** The SQLite database (`~/.claude/dream.db` by default), suggestion previews (`.suggestions/`), and the memory store remain on your machine.
- **Check your configuration first.** Before the first run, review the `[[sources]]` roots and the `[stages.*]`/`[providers.*]` entries in `~/.config/dream/config.toml` so you know which transcripts are read and where their content is sent.
- **Local and external effects.** `estimate`, `search`, `status`, `preflight`, `context`, and `suggestions list` only read the persistent local data. Automatic recall opens SQLite through an immutable read-only connection; its small deduplication markers live under `/tmp`. `ingest` reads transcripts and updates the local SQLite index without calling a model. `distill` and `consolidate` make model calls and write local results. Suggestion review commands update local review state; `accept`, `merge`, and configured apply operations can also update memory files, with backups before live-memory writes.
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
claude-jsonl       codex/claude    conflict-safe apply
```

The pipeline stages are:

1. `ingest`: discover configured sources, filter transcripts, index messages with SQLite FTS5.
2. `distill`: call the provider configured for the `distill` stage once per changed session revision.
3. `consolidate`: call the configured `consolidate` provider and produce suggestions.
4. `suggestions`: inspect and resolve suggestions with file-version preconditions and backups.

Consolidation uses an explicit ledger of consumed distillation identities. It does not use session timestamps as its progress cursor and does not silently consume truncated batches.

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

Automatic recall is disabled at the Codex hook boundary until `dream hooks install`
has been reviewed and trusted through Codex `/hooks`. The hook is read-only with
respect to `~/.claude/dream.db`, so it works in Codex's workspace sandbox without
extra launch flags or writable-home configuration. The host-side systemd timer
remains the write path for ingest, distillation, consolidation, and memory review.
The lexical store is local, raw transcripts are disabled by default, and rendered
results are marked as untrusted reference data with source provenance. `recall.diagnostic_path` receives
bounded JSONL diagnostics; diagnostics never contain raw document bodies.

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
dream suggestions accept 12
dream suggestions reject 13
dream suggestions merge 14 --body-file /tmp/merged.md
dream suggestions apply-configured
```

`accept` and `merge` refuse to write when the target changed after suggestion creation. `apply-configured` applies all pending suggestions only when `review.mode = "auto-apply"`; in `suggest-only` mode it exits successfully without writing.

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
dream suggestions apply-configured
```

Provider routing and review mode come from TOML; the unit contains no provider/model selection.

```bash
cp dream/systemd/dream.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now dream.timer
```

On macOS, schedule the same commands with `launchd` or cron.

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
