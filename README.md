# dream — provider-agnostic local memory consolidation

`dream` ingests agent transcripts, filters them to visible user/assistant turns, distils sessions into structured findings, and consolidates those findings into reviewable persistent-memory suggestions.

Transcript sources and model providers are independent. The packaged profile currently reads Codex JSONL and runs both model stages through `codex exec`; Claude JSONL and Claude CLI remain optional adapters.

## Architecture

```text
Codex skill / shell / systemd
                         |
                    neutral CLI
                         |
        +----------------+----------------+
        |                |                |
 transcript sources   stage router    SQLite + memory
        |                |                |
 codex-jsonl         model provider    suggestions
 claude-jsonl        codex/claude      conflict-safe apply
```

The pipeline stages are:

1. `ingest`: discover configured sources, filter transcripts, index messages with SQLite FTS5.
2. `distill`: call the provider configured for the `distill` stage once per changed session revision.
3. `consolidate`: call the configured `consolidate` provider and produce suggestions.
4. `suggestions`: inspect and resolve suggestions with file-version preconditions and backups.

Consolidation uses an explicit ledger of consumed distillation identities. It does not use session timestamps as its progress cursor and does not silently consume truncated batches.

## Configuration

The default path is `~/.config/dream/config.toml`. `install.sh` creates it from `dream/default-config.toml` only when it does not already exist.

```toml
[stages.distill]
provider = "codex"
model = "gpt-5.6-luna"
reasoning_effort = "low"

[stages.consolidate]
provider = "codex"
model = "gpt-5.6-sol"
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

Model names belong to configuration, not the core. Configuration precedence is CLI override, stage-specific environment variable, TOML, then packaged defaults. Unknown provider/source types and unknown options fail explicitly.

Changing provider/model configuration does not automatically re-run every historical session. Use `dream distill --refresh` for an explicit provenance refresh; changed transcript revisions are always re-distilled.

Legacy compatibility variables remain accepted: `DREAM_BACKEND`, `CLAUDE_DREAM_DB`, and `DREAM_MEMORY_ROOT`.

## Install

Run the installer from the checkout:

```bash
./install.sh
```

The installer links:

- `dream/dream.py` to `~/.local/bin/dream`;
- `skill/` to `~/.codex/skills/dream`;
- and installs `dream/default-config.toml` to `~/.config/dream/config.toml` only when no configuration file exists.

Optional providers are loaded lazily. A missing `claude` executable does not affect Codex-backed stages, ingest, status, or search.

## Layout

- `dream/` — the Python runtime: CLI, tests, configuration defaults, prompts, and systemd units.
- `skill/` — the Codex skill: `SKILL.md` and `agents/openai.yaml`.

## Testing

pytest is a development dependency. Run the suite from the repository root:

```bash
pytest -q dream
```

## Usage

```bash
dream ingest
dream estimate -v
dream distill --yes
dream consolidate
dream status
dream preflight
```

Search indexed conversations:

```bash
dream search "postgres migration"
dream search "docker daemon ip" --role assistant
dream search 'NEAR("dream" "memory", 5)' --project -home-alice-Dev-api
```

Machine-readable review:

```bash
dream suggestions list --json
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
- memory: `~/.claude/projects/<home-slug>/memory/`.

These are storage locations, not runtime dependencies on Claude Code. Schema upgrades are additive, versioned, and create a consistent SQLite backup before modifying a legacy on-disk database.

Each session records source, external identity, content revision, and parser version. Each distillation records source revision, prompt version, provider, model, usage when available, and duration.

## Scheduling

`systemd/dream.service` runs:

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
