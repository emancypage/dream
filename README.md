# dream — provider-agnostic local memory consolidation

`dream` ingests agent transcripts, filters them to visible user/assistant turns, distils sessions into structured findings, and consolidates those findings into reviewable persistent-memory suggestions.

Transcript sources and model providers are independent. The packaged profile reads Codex JSONL and Open Code SQLite sessions and runs both model stages through `codex exec`; Claude JSONL and Claude CLI remain optional adapters.

For a visual walkthrough of one nightly run, see [How Dream works](https://emancypage.github.io/dream/how-dream-works.html). The page steps through each stage on a sleep hypnogram. Its source is [`docs/how-dream-works.html`](docs/how-dream-works.html).

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

### Recall and consolidation settings

The packaged defaults keep the earlier behaviour: the embedder is off, usage
does not reorder results, consolidation sends the whole store, and no local
model provider is configured. `dream preflight` reports the embedder, whether
the MCP server entry is in the Codex or OpenCode configuration (Claude Code
keeps its own: `claude mcp get dream`) and the consolidation mode.

| Key | Default | Meaning |
|---|---|---|
| `recall.usage_tie_break` | `false` | Use harvested recall usage to break exact ranking ties (after trust). Harvesting and audit annotations run either way. Turn it on only after a calibrated eval shows no regression. |
| `recall.embedder.enabled` | `false` | Turn on the dense candidate list. |
| `recall.embedder.type` | `"none"` | `"openai-compatible"` is the only other value: a local embedding server such as `llama-server --embedding` or Ollama. |
| `recall.embedder.remote_data_egress` | `false` | Must be `true` for a `base_url` that is not loopback (`127.0.0.0/8`, `::1` or `localhost`); otherwise the embedder is a configuration error and recall stays lexical. |
| `recall.embedder.base_url` | `"http://127.0.0.1:8081"` | `http` or `https`; requests go to `/v1/embeddings`. |
| `recall.embedder.model` | `"bge-m3"` | Model name sent to the server; part of the vector fingerprint. A placeholder until the private eval picks a model. |
| `recall.embedder.dimension` | `1024` | Vector length, 1 to 4096; every response must match it. Part of the fingerprint. |
| `recall.embedder.timeout_seconds` | `1.5` | Wall-clock bound on one query embedding, at most 3 s (hooks time out after 5 s). |
| `recall.embedder.dense_candidates` | `200` | Length of the dense list fused with the lexical lists. |
| `recall.embedder.api_key_env` | unset | Name of an environment variable holding a bearer token; the value is never stored or logged. |
| `consolidate.mode` | `"full"` | `"full"` sends the whole store nightly, byte-identical to earlier releases. `"subset"` sends `MEMORY.md` plus related topic files and leaves the staleness audit to the weekly `dream audit`. `dream consolidate --mode` overrides it for one run. |
| `consolidate.subset_files` | `15` | Topic files sent in subset mode, besides `MEMORY.md` and files whose name a finding mentions. |
| `stages.audit` | `provider = "codex"`, `reasoning_effort = "high"`, `timeout_seconds = 1800` | The model stage of `dream audit`, with the same keys as every stage. |
| `providers.<name>.type = "openai-compatible"` | not configured | A local OpenAI-compatible generation server (llama-server, Ollama, vLLM) that a stage can route to. Keys: `base_url` (required), `model`, `api_key_env`, `remote_data_egress` (default `false`, same loopback rule) and `timeout_seconds` (default 600). `default-config.toml` has a commented `[providers.local-qwen]` example. |

Embedder problems never break recall or ingest: an invalid `[recall.embedder]`
table disables the embedder, recall stays lexical, and the `recall.embedder`
preflight check and the recall diagnostics name the problem. The dense list reaches hooks and agent queries only after
`dream recall-calibrate --mode lexical_plus_embedder --apply` has stored a
threshold for the current model and dimension.

## Privacy and data flow

- **Local reads.** `dream ingest` and `dream search` read only local transcript stores (by default Codex JSONL under `~/.codex/sessions` and Open Code SQLite at `~/.local/share/opencode/opencode.db`) and the local Dream SQLite database. By default they make no network or model calls. When `[recall.embedder]` is enabled, `dream ingest` (like `dream embed`, `dream recall-eval` and `dream recall-calibrate --mode lexical_plus_embedder`) also sends scrubbed document text to the configured embedding server, loopback only unless `remote_data_egress = true`; `dream search` never transmits anything.
- **Model calls send content.** The `distill` and `consolidate` stages send selected session content to the model provider configured for that stage — by default the Codex CLI against your ChatGPT subscription. Distillation sends the filtered transcript of a session; consolidation sends the distilled notes plus the current memory store. The review stage uses its configured provider and sends pending suggestions plus bounded target context.
- **Local embedder (off by default).** With `[recall.embedder]` enabled, `dream ingest` and `dream embed` send the scrubbed text of approved memories and distilled summaries (raw transcripts only with `recall.allow_raw_transcript_prompt = true`) to the embedding server, and prompt hooks and agent queries send the scrubbed query. `dream recall-eval` and `dream recall-calibrate --mode lexical_plus_embedder` send the scrubbed text of every fixture document and query, which for the private fixture are copies of your memories. Only loopback URLs are accepted unless `remote_data_egress = true`. Vectors stay in the local database.
- **Local generation provider.** A stage routed to an `openai-compatible` provider sends that stage's prompt to its `base_url`, again loopback only unless `remote_data_egress = true`. For `distill` that prompt holds the filtered session text.
- **Private eval fixture.** `dream eval-build` sends sampled approved memories and summaries, with the retention policy, to the configured `distill` provider to write test questions, and stores copies of those memories in `~/.config/dream/eval/private.json` (file mode 0600, directory 0700), outside the repository. `.gitignore` also excludes `dream/fixtures/private*`. `dream recall-calibrate` refuses public fixtures and anything under `dream/fixtures/`, evaluates on an in-memory copy, and writes to the live database only with `--apply`.
- **Weekly audit and subset mode.** `dream audit` sends every topic file, in chunks of at most 200,000 characters (a larger file gets a chunk of its own), with the `MEMORY.md` index and per-file usage annotations, to the `audit` stage provider. Subset consolidation sends `MEMORY.md`, the remaining index lines and the related topic files (at most `subset_files`, plus files a finding names) instead of the whole store.
- **Recall usage.** `dream ingest` counts which documents the hooks selected, from hook markers older than 24 hours, into the local `recall_usage` table and deletes those markers. `dream recall` and agent queries are never counted. Usage only annotates and breaks ties; it never deletes anything.
- **Everything else stays local.** The SQLite database (`~/.claude/dream.db` by default), suggestion previews (`.suggestions/`), and the memory store remain on your machine.
- **Check your configuration first.** Before the first run, review the `[[sources]]` roots and the `[stages.*]`/`[providers.*]` entries in `~/.config/dream/config.toml` so you know which transcripts are read and where their content is sent.
- **Local and external effects.** `estimate`, `search`, `status`, `preflight`, `context`, `recall`, `mcp serve`, and `suggestions list` only read the persistent local data. Automatic recall, `dream recall` and the MCP server open SQLite through an immutable read-only connection; the hooks' small deduplication markers live under `/tmp`, and `dream recall` and the MCP server leave none. `mcp install` and `mcp uninstall` edit one client's configuration file, with a backup. `ingest` reads transcripts and updates the local SQLite index; by default it makes no network or model calls, and with `[recall.embedder]` enabled it sends scrubbed document text to the configured embedding server (loopback only unless `remote_data_egress = true`). `distill`, `consolidate`, and configured curation make model calls and write local results. Suggestion review commands update local review state; `accept`, `merge`, and configured apply operations can also update memory files, with backups before live-memory writes.
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

1. `ingest`: count recall usage from old hook markers, discover configured sources (Codex JSONL, Open Code SQLite, optional Claude JSONL), filter to visible user/assistant turns, and index messages with SQLite FTS5. It then synchronizes the recall index and, when the local embedder is enabled, embeds new or changed recall documents.
2. `distill`: call the provider configured for the `distill` stage once per changed session revision.
3. `consolidate`: call the configured `consolidate` provider and produce suggestions, from the whole store (`full`) or from `MEMORY.md` plus related files (`subset`, which can also propose exact-span `patch` edits).
4. `suggestions`: inspect and resolve suggestions with file-version preconditions and backups.
5. `audit` (weekly, optional): review the whole store in chunks for stale files and propose only `remove` and `patch` suggestions. Needed only with `consolidate.mode = "subset"`.

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

Three latency tests carry the performance budgets: session-start recall p95
at most 150 ms and prompt recall p95 at most 250 ms on 10,000 synthetic
documents, and a dense scan p95 at most 400 ms over 5,000 vectors of 1,024
dimensions. Those budgets at scale 1 are the acceptance gates. The tests
multiply them by `DREAM_PERF_BUDGET_SCALE`, which defaults to `3` when the
`CI` environment variable is set and to `1` otherwise; a larger scale is only
a regression check for slow shared runners, never the acceptance budget. Each
test prints `perf: <name> p95=<ms> budget=<ms> scale=<x>` and puts the measured
p95 and the effective budget in its failure message, so
`pytest -s -p no:cacheprovider dream -k "latency_budget or dense_scan_p95"` shows the raw
numbers.

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
dream recall "postgres migration rollback"
dream recall "release checklist" --project api --kinds approved_memory --limit 3 --json
dream mcp install --client codex
dream mcp uninstall --client codex
dream recall-eval --fixtures dream/fixtures/recall/public.json
dream eval-build --limit 100
dream recall-eval --fixtures ~/.config/dream/eval/private.json --by-language
dream recall-calibrate --fixtures ~/.config/dream/eval/private.json --mode lexical --apply
dream embed
dream embed --refresh
dream consolidate --mode subset
dream audit
dream status --usage
```

### Recall evaluation, embeddings and the weekly audit

- `dream eval-build [--limit N] [--languages en,pl] [--questions-per-file 3] [--output PATH] [--yes]` builds the private held-out fixture from the live recall index. It makes one `distill` model call per sampled document; without `--limit` it needs `--yes`.
- `dream recall-eval --fixtures PATH --by-language` adds Recall@1, Recall@5, MRR, forbidden results and p95 latency for each mode (`lexical`, plus `lexical_plus_embedder` when the embedder is on) and language (`en`, `pl`, `all`). It runs on an in-memory copy and never opens the live database.
- `dream recall-calibrate --fixtures PATH --mode MODE [--output FILE] [--apply]` computes the selection threshold for one mode. Without `--apply` it only prints the record (and writes `--output`); `--apply` is the only writer of `recall_calibrations`. Rerun it with `--mode lexical_plus_embedder` after changing the embedder model or dimension and after `dream embed --refresh`.
- `dream embed [--refresh]` synchronizes the recall index, then stores vectors for documents that have none under the current embedder (`--refresh` re-embeds everything, the step after swapping a model behind the same name). Nightly ingest does the incremental part on its own.
- `dream consolidate --mode full|subset` overrides `consolidate.mode` for one run.
- `dream audit [--force] [--model MODEL]` runs the weekly staleness audit. With `consolidate.mode = "full"` it exits without a model call unless `--force` is given, because the nightly run already audits the whole store.
- `dream status --usage` lists approved memory files that automatic recall has not selected in the last 90 days, with their index lines and the start of the observation window. It only reads.

Agents can also query approved memory during a session with `dream recall` or the MCP server; see [Mid-session recall for agents](#mid-session-recall-for-agents).

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

Optional `recall.embedder` and `recall.reranker` adapters are disabled by default
and fall back to lexical results on unavailable adapters, invalid output, missing
calibration, or adapter errors. The embedder talks to a local OpenAI-compatible
server through the standard library. Its vectors are stored per content hash and
fingerprint (`openai-compatible:<model>:<dimension>`), and the nearest ones join
the four lexical lists (FTS, structure, paths, recency) as a fifth reciprocal-rank
list, on the prompt path only. Hooks make at most one embedding request (the
query) and never embed documents. Set `remote_data_egress = true` only after
reviewing the adapter's data-flow implications; Dream does not add a model package
or network dependency.

On stores with more than about 500 eligible documents, session start
deliberately considers only the FTS matches plus the 500 most recent
documents for structure and path evidence; smaller stores rank exactly as a
full scan would. Excerpts are scrubbed only when they are read. The
synchronized `path_text` column holds a normalized copy of each document's
project slug, source path and full body; like the `text` column it is not
scrubbed.
`dream/test_recall_query.py` holds the session-start path to 150 ms and the
prompt path to 250 ms at p95 on a synthetic 10,000-document store, scaled on
CI as described under Testing.

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

## Mid-session recall for agents

Hooks inject memory at session start and on the first prompt. During a
session an agent can ask for more in two read-only ways; both use the same
retrieval as the prompt hook (the `agent-query` event, with its calibrated
threshold) and render every result through the same untrusted-reference
marker, provenance line and secret scrubber.

`dream recall "<3-8 content words>"` prints the rendered recall block, or
nothing when no memory passes the threshold. `--project SLUG` prefers one
project without excluding the others, `--kinds` filters source kinds
(default `approved_memory,distilled_summary`; `raw_transcript` must be named
and needs `recall.allow_raw_transcript_prompt = true`), `--limit` is clamped to 1-10
(default 5) and `--json` adds ids, kinds, trust, project, scores, mode and the
calibration version. A missing or busy database, an unmigrated schema or
disabled recall print one `dream recall: unavailable (...)` line to stderr and
exit 0.

`dream mcp serve` exposes the same data as Model Context Protocol tools over
stdio (JSON-RPC 2.0, one message per line; protocol versions 2025-06-18,
2025-03-26 and 2024-11-05; only the `tools` capability):

| Tool | Arguments | Result |
|---|---|---|
| `recall_search` | `query`, optional `project`, `kinds` (same default as `dream recall`), `limit` (clamped to 1-10) | The rendered recall block, as `dream recall` prints it, plus structured `results` rows of `{id, kind, trust, project, score}` |
| `recall_get` | `id` from a `[source: <id>; ...]` line | That one document, if retention policy, the constitution guard and the raw-transcript opt-in allow it, rendered with excerpts up to `recall.prompt_budget_codepoints` |
| `memory_index` | none | The indexed `MEMORY.md`, rendered within `recall.session_start_budget_codepoints`, and the number of other approved memory documents |

The server is meant to run inside a sandboxed agent, so it is read-only. Each
tool call opens the database with an immutable read-only connection and closes
it before replying, so a long session still sees the nightly updates. A
successful call writes nothing: no database rows, hook markers, embedding
cache or files under the home directory. Memory text comes only from the
database (`memory_index` serves the synchronized copy of `MEMORY.md`); the one
file read under the memory root is `.dream-retention.json`, so exclusions apply
even to documents synchronized before the policy changed. A failed call may append one bounded diagnostic to `recall.diagnostic_path`
(once per tool and error in a session, never for an active WAL), and a failed
append is ignored. Structured results also carry the rendered block as
`context`, so a client that forwards only structured content still passes the
marked, scrubbed text. Tool failures (database missing or busy,
schema not migrated, unknown or excluded id) come back as results with
`isError: true`; malformed JSON and lines over 1 MiB get `-32700`, batches
(JSON arrays) and invalid requests `-32600`, unknown methods `-32601`, unknown
tools or invalid arguments `-32602`. stdout carries only protocol messages.

The server is not installed by default. `dream mcp install --client codex`
adds a marker-delimited `[mcp_servers.dream]` block at the end of
`$CODEX_HOME/config.toml` (default `~/.codex/config.toml`); `--client opencode`
adds `mcp.dream` (`{"type": "local", "command": [<dream>, "mcp", "serve"],
"enabled": true}`) to `$OPENCODE_CONFIG` or the global
`~/.config/opencode/opencode.json` (`opencode.jsonc` when only that exists;
it must be plain JSON, and an empty file counts as `{}`); `--client claude`
runs `claude mcp add --scope user dream -- <dream> mcp serve`, or prints that
command when `claude` is not on `PATH`. `<dream>` is the absolute path of the
`dream` launcher found on `PATH`; without one the entry runs `dream` by name
and the client must find it on its own `PATH`. When `DREAM_CONFIG`,
`DREAM_MEMORY_ROOT` or `CLAUDE_DREAM_DB` is set, install records it in the
entry's environment (Codex `env`, OpenCode `environment`, Claude `-e`) and
prints the names, not the values. `--path` points codex or opencode at another
file.

Install refuses malformed files, foreign `dream` entries, `[[mcp_servers]]`
arrays, read-only files and dangling symlinks, re-parses the result before an atomic replace,
keeps unrelated entries, line endings, a UTF-8 byte order mark and the JSON
indentation, writes a timestamped `.bak-` copy of the previous file, and writes
nothing when the entry is already current. A changed Codex block is reported as
updated; an existing OpenCode entry keeps the keys you added and an explicit
`"enabled": false`. For Claude Code, an existing `dream` server counts as
installed only when `claude mcp get dream` shows `mcp serve` (it is kept as
is: uninstall and install again to refresh it); every other
`claude` failure is reported with its output. Claude Code keeps its own file,
so Dream makes no backup there. `dream mcp uninstall --client ...` removes only
the Dream entry and is the rollback; for Codex it restores the file byte for
byte. Neither command starts the server: restart the client and check its MCP
list (`/mcp` in Codex and Claude Code, `opencode mcp list`).

## Data and compatibility

Existing paths remain the defaults during migration:

- DB: `~/.claude/dream.db`;
- memory: `~/.claude/projects/<home-slug>/memory/` (the packaged default resolves the home slug for the running user).

These are storage locations, not runtime dependencies on Claude Code. Schema upgrades are additive and create a consistent SQLite backup before modifying a legacy on-disk database. The recall upgrade's columns and tables (embedding blobs, recall usage, patch spans, stored path text) are detected by their shape rather than by a new version row, and only a writing command migrates them: hooks and other read-only commands work unchanged on a database that has not been migrated yet.

Each session records source, external identity, content revision, and parser version. Each distillation records source revision, prompt version, provider, model, usage when available, and duration.

## Scheduling (Linux)

`dream/systemd/dream.service` runs:

```text
dream ingest
dream distill --yes --limit 50
dream consolidate
dream suggestions curate-configured
dream character generate --source all --limit 10
dream character judge --limit 10
```

The review stage asks its configured provider for accept/reject/merge/defer decisions. The unit contains no literal provider commands. TOML selects all providers and models, including Claude-backed review when configured.

```bash
./install.sh
systemctl --user daemon-reload
systemctl --user enable --now dream.timer
```

`dream ingest` also counts recall usage and, when the embedder is enabled,
embeds new or changed recall documents, so the nightly unit needs no extra stage.

The weekly staleness audit has its own units. `install.sh` copies
`dream-audit.service` and `dream-audit.timer` (Sundays at 12:00,
`Persistent=true`, up to 15 minutes of random delay) but enables nothing. The
service runs `dream audit` directly, outside the nightly run metrics. Enable it
when you switch to `consolidate.mode = "subset"`:

```bash
systemctl --user enable --now dream-audit.timer
```

On macOS, schedule the same commands with `launchd` or cron.

## Monitoring (Linux)

The scheduled service records a bounded JSON state file at
`~/.local/state/dream/run.json`. Each stage is wrapped by `dream-stage`; a
failure preserves the failing stage and a safe error code. The separate
`dream-exporter.service` exposes only numeric Prometheus metrics and never
exports provider error text or transcript contents.

The exporter should bind to an address reachable by Prometheus but not to every
network interface. Configure it to listen on the private address reachable by monitoring. The main metric is `dream_run_health`: `3` = success,
`2` = running, `1` = stale, `0` = failed, and `-1` = no recorded run.

The test-backed patchers add the exporter target and five panels to the existing
fleet dashboard:

```bash
python3 dream/monitoring_prometheus.py /path/to/prometheus.yml MONITOR_ADDRESS:9409
python3 dream/monitoring_dashboard.py /path/to/rpi-fleet.json
```

Both patchers are idempotent, reject ambiguous targets, and preserve unrelated
dashboard panels. Back up the live files before applying generated output.

`dream_recall_documents_unused_90d` counts approved memory documents that automatic
recall has not selected in the last 90 days. Like every exporter metric it carries
no document IDs or paths.

The short-review panel shows `dream_character_short_drafts_pending`. The exporter
reads the current eligible draft count on every scrape, so approval updates the
view without waiting for a nightly run. It emits no draft text, IDs, or source
paths. `dream_character_short_review_source_up` is `1` when that read succeeds.
If it fails, the count is omitted and the health signal is `0`. The dashboard
requires both source health and exporter availability, and shows unavailable data
instead of a false zero. Waiting drafts are orange; zero is green.

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
| `mcp_server.py`, `mcp_install.py` | read-only MCP stdio server and client configuration installers |
| `recall_eval.py`, `recall_adapters.py` | fixture evaluation, calibration, and optional adapter contracts |
| `eval_build.py` | private held-out recall fixture builder |
| `recall_embeddings.py` | float32 vector store and the dense candidate scan |
| `recall_usage.py` | recall usage harvest, tie-break map and unused-memory report |
| `metrics.py`, `stage_runner.py` | run-state persistence and Prometheus exporter |
| `monitoring_prometheus.py`, `monitoring_dashboard.py` | idempotent Prometheus and Grafana patchers |

## Character pipeline

Character is disabled by default. It builds preference pairs and deploys approved
examples through context. It never changes model weights. Memory suggestion
curation cannot accept character pairs. Each pair needs an explicit human decision.

The installer creates a draft constitution at
`~/.config/dream/character/constitution.md` only if it does not exist. Edit that
file yourself. Keep the trait IDs stable and fill each trait's `Why:` line, the
priority order, and hard constraints. Seed prompts are optional. Remove unused
trait TODO slots before approval. Set `status` and `approved_at` yourself when you
approve the document. Dream never approves or edits the live constitution.

Configure the new routes in your local TOML. Packaged stage defaults select only
providers; set explicit model IDs locally before enabling model calls.

```toml
[character]
enabled = false
constitution_path = "~/.config/dream/character/constitution.md"
state_root = "~/.local/state/dream/character"
allow_same_family_judge = false
hook_injection = false
render_max_pairs = 3
render_budget_codepoints = 12000
max_context_codepoints = 12000
max_response_codepoints = 8000
max_seed_expansions = 0

[stages.character_extract]
provider = "codex"

[stages.character_generate]
provider = "codex"

[stages.character_judge]
provider = "claude"

[stages.character_eval_response]
provider = "codex"
```

The judge family must differ from both extraction and regeneration families.
Codex with ChatGPT subscription auth is OpenAI; Claude CLI is Anthropic. The
same-family override is explicit and is recorded in provenance. Provider checks
are scoped to the command. Disabled character stages need no model or executable.
Historical Claude-session responses carry an overlap warning when Claude judges
them. The original response can benefit from that shared-family bias. Review must
show the warning, including verdicts against the proposed chosen response.

```bash
dream character estimate --source all --limit 10
dream character generate --source all --limit 10
dream character judge --limit 10
dream character list
dream character show PAIR_ID
dream character accept PAIR_ID --expected-sha HASH --reason "Reviewed against the constitution"
dream character accept PAIR_ID --answer right --expected-sha HASH --reason "Prefer the historical answer" --acknowledge-flags
dream character defer PAIR_ID --expected-sha HASH
dream character reject PAIR_ID --expected-sha HASH --reason "The answer lacks evidence"
dream character edit PAIR_ID --expected-sha HASH --body-file /tmp/chosen.txt --reason "Corrected the example"
dream character short draft PAIR_ID --expected-pair-sha HASH --body-file /tmp/short.json
dream character short list
dream character short show SHORT_ID
dream character short generate PAIR_ID --expected-pair-sha HASH
dream character short approve SHORT_ID --expected-sha SHORT_HASH --reason "Reviewed the exact short wording"
dream character short reject SHORT_ID --expected-sha SHORT_HASH --reason "This changes my meaning"
dream character render --preview
dream character render
dream character export --format dpo-jsonl --preview
dream character export --format dpo-jsonl
```

Enabled database-writing commands initialize the new schema with a backup before
model work. Normal ingestion also initializes it. Read-only commands report a
missing schema without migrating it. Render and export also keep SQLite read-only,
including when saving files. Copy the pair hash from `show`. Editing invalidates judgment and acceptance, so
judge it again before accepting. Conflict, inconsistent judgment, or a verdict
against chosen requires `--acknowledge-flags` and a human reason. Generation
`--refresh` creates a new attempt and keeps old review history. Synthetic generation
skips traits without seeds without treating that as a failure. Seed expansion is
explicit through `--expand-seeds` and bounded by `max_seed_expansions`.

Short examples use `dream-character-short-v1` in a separate database table. The
input JSON contains five nonempty text fields: `situation`, `preferred`, `avoid`,
`feedback`, and `limits`. Their combined length is at most 2500 Unicode codepoints.
Draft them only from accepted pairs. The full pair, answers, judgments, and prior
review records stay unchanged. The source pair ID, revision, and hash are recorded
automatically. Full-record DPO export continues to use the full accepted answers.

When `character.enabled = true`, accepting a full pair automatically requests a
short draft through the configured `character_generate` provider and model. The
model receives the conversation, the selected answer, the other answer, and the
human's acceptance reason. Right-answer choices use the historical answer as the
preferred text. The draft records its model and prompt identity and remains
unapproved until a separate human review. When character generation is disabled,
acceptance still works and reports that short drafting was skipped.

Acceptance is saved before the model call. If drafting fails, the command returns
a nonzero exit code with the accepted pair and the failure in its JSON output.
Retry only the draft with `short generate`, using the accepted pair's current hash.
This command also prepares drafts for already accepted pairs. It leaves an existing
short version for the same source unchanged. A new source revision needs a new
draft and human approval. Model failures, invalid output, and source or policy
changes cannot approve a draft or undo the full-pair decision.

Show the complete short wording and hash to the human before approving it. An
accepted full pair does not approve its short version. `short approve` records a
separate human decision bound to the exact short hash. To change a short version,
run `short draft` again with its current `--expected-sha` and the current source
`--expected-pair-sha`. Changing the wording returns it to draft status. Source or
constitution changes make it stale. Policy exclusions apply through source lineage.
Manual drafting and short review make no model calls. Database migration adds the short-example table
after backing up the existing database.

`render` writes a character block and manifest under the state directory. It uses
the complete approved constitution and at most three approved short examples. It counts
the header and complete examples within the codepoint budget. An oversized
constitution causes an error. Dream never installs the block into `AGENTS.md`,
`CLAUDE.md`, or native Codex Memories. You choose where to load it. Copies outside
Dream's state directory must be replaced manually after a constitution or policy
change.

Setting `hook_injection = true` adds a separate trusted character section to
session-start context. Recall stays a separate untrusted section with relevance
ranking. The hook makes no model calls and writes no database state. Its total
context limit can be smaller than the file-rendering budget. Session startup renders
approved short examples directly from the database. No saved render is required.
Draft, rejected, stale, or unreviewed short versions are never injected. When no
short version is approved, the hook includes the constitution without examples.
A character block
which cannot fit is omitted with a diagnostic; the constitution is never cut.
The flag defaults to false. Review any changed hook definition before trusting it.
Injection is independent of `character.enabled`, which controls generation,
judging, and model-based evaluation. Set `hook_injection = true` and
`enabled = false` to use an approved constitution in sessions while those model
stages stay disabled.

When current short drafts are waiting for review, the session-start hook adds a
count-only reminder and the command `dream character short list`. This notice is
independent of `character.hook_injection`; it works while example injection or
model generation is disabled, provided automatic recall is enabled. It never
includes draft wording or approves anything. Approved, rejected, stale, and
policy-excluded drafts do not contribute to the waiting count. There is no notice
when the count is zero and no reminder on individual prompt submissions. The
notice shares the hook's context budget and takes priority over an oversized
character block. Failure to read the queue does not block normal recall.

Session-start context defaults to an 18,000-byte budget when character injection
is enabled. This budget holds the complete character block and recalled memory.
The default character rendering budget is 12,000 codepoints. Plain English
character blocks of that size leave about 6,000 bytes for recall; other scripts
can require more bytes. A character block that exceeds the combined budget is
omitted in full. Both `recall.session_start_budget_codepoints` and
`recall.session_start_additional_context_limit` default to `18000`, and generated
Codex session-start hooks use `additionalContextLimit = 18000`. Explicit local
configuration overrides the packaged defaults. Prompt-recall budgets are
unchanged.

Character data flow is explicit. Extraction sends a bounded visible task window,
the original assistant response, the following correction, and the constitution
to the extraction provider. Regeneration sends the bounded original task window
and constitution, without the later correction. Synthetic chosen and baseline
responses use the same model; only chosen receives the constitution. Judge calls
send the pair and constitution twice with swapped response order. All model calls
receive the local retention policy. Estimate makes no model calls and reports item
and call bounds before generation. These calls consume provider usage.

```bash
dream character eval --fixtures dream/fixtures/character/public.json --limit 10
dream character eval --report RUN_ID
```

Evaluation compares baseline with the rendered block on fixed multi-turn probes
and neutral task controls. It uses an independent-family judge and swapped order.
Reports include failed, unsupported, and inconsistent samples. A trait-specific
probe must match its trait text hash; reusing an ID for a different trait does not
silently reuse its score. Public probes never become generation seeds. Evaluation sends fixed probes and the rendered examples to the response provider,
and both trajectories plus the constitution to the judge. Reports
stay local until you choose to publish them. Context deployment does not prove
the same robustness as weight training.

DPO export uses the versioned `dream-dpo-v1` format. Each JSONL row has `prompt`,
`chosen`, `rejected`, and `metadata`, including constitution identity, human review,
and source provenance. Correction prompts are bounded visible-context
reconstructions; original tool and system context may be unavailable. Export
includes accepted current-policy pairs only. There is no trainer, model download,
LoRA implementation, or endpoint in this release.

Retention applies before model calls, writes, and reads. `dream retention` previews
affected derived character rows and managed files. `--apply` includes them in the
same `affected-records.json` backup as other affected memories, then deletes the
rows and their lineage dependents. It never changes human seeds in the constitution
or original transcripts. A changed affected set aborts the character cleanup and
requires a rerun. User-modified managed files stay in place and are reported.
A retention conflict inside the constitution blocks character use until you edit
the human-owned document. Private backups remain historical copies outside recall.

The nightly service adds `character_generate` and `character_judge` after memory
curation, each with a limit of 10. Each stage has separate failure metrics and
continues after earlier stage failures. Both skip cleanly while character is
disabled. The service contains no literal provider commands; configuration selects
the providers. Human review, rendering, export, and evaluation never run nightly.
