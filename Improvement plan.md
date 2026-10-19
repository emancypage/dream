# Dream improvement plan for agentic sessions

Oct 19, 2026 · @Szymon Szymkowiak

> For agentic workers: use superpowers:subagent-driven-development or superpowers:executing-plans. Follow the test-first steps below. Export this tab as Markdown to `docs/superpowers/plans/2026-10-19-recall-and-consolidation-upgrade.md` and commit it before starting Task 0.

## Goal

Close the four measured gaps from the review tab without giving up what `dream` already does right: cross-lingual hybrid recall, mid-session memory access for every agent, incremental consolidation with fact-level patches, and usage-aware forgetting, all gated by a private held-out eval built from real sessions. The review gate, provenance, read-only hooks and stdlib-only runtime stay exactly as they are.

## Architecture

Recall gains a dense candidate list beside the four lexical lists in the existing RRF fusion; vectors come from a local OpenAI-compatible embedding server (llama-server or Ollama on localhost) through `urllib`, stored as float32 blobs in `recall_embeddings`. A new `dream recall` command and a stdio MCP server in `dream/mcp_server.py` expose the same read-only `run_context` path to Codex, OpenCode and Claude Code. Consolidation reads MEMORY.md plus the top-k memory files retrieved for the batch instead of the whole store, and can emit a `patch` suggestion kind applied under the existing SHA preconditions. A nightly sweep harvests selected-document markers into a `recall_usage` table that ranking tie-breaks and the staleness audit consult. An optional `openai-compatible` generation provider lets `distill` run on the local Qwen executor.

## Tech stack

Python 3.11+ standard library (`urllib.request`, `array`, `struct`, `json`, `sqlite3`), TOML, pytest with fake HTTP servers (`http.server` on a loopback port) and fake providers. No numpy, no sqlite-vec, no model packages in the runtime.

## Spec

The review tab of this doc is the design basis. Task 0 turns it into `docs/superpowers/specs/2026-10-19-recall-and-consolidation-upgrade-design.md` for approval before any code changes.

## Global constraints

- Work on `feat/recall-upgrade` from fresh `main`. Preserve untracked `session-logs/` files.
- No local configuration edits, real model calls, pushes or PRs. Tests use fake providers and a fake loopback HTTP server.
- Runtime stays Python 3.11+ standard library. pytest remains a development dependency only.
- Hooks and the MCP server stay read-only: `storage.open_db_readonly`, immutable URI, no WAL creation, no schema migration. Every failure exits 0 with no stdout on the hook path and a JSON-RPC error on the MCP path.
- Every new recall path renders through `render_context`: the untrusted-reference marker, provenance line and scrubber are mandatory. No new code may emit a raw document body.
- New config keys default to the current behaviour: embedder off, MCP server not installed, consolidation subset off (`consolidate.mode = "full"`), usage harvesting on but non-ranking until calibrated.
- Human approval stays mandatory for every memory write. `patch` suggestions obey the same SHA preconditions, protected-file checks and backups as `update`.
- The character pipeline is not touched. Any task that needs a change there stops and reports.
- Run `python3 -m pytest -q -p no:cacheprovider dream` and `git diff --check` before every commit. Short conventional commit subjects (`feat:`, `fix:`, `test:`, `docs:`).
- Each task ends with a commit. A task that cannot meet its tests leaves no partial commit.

## Review focus

- A dense candidate list must not become a channel for secrets or injection: Task 2 tests that dense-only candidates pass the same scrubber and marker as lexical ones, and that a document excluded by retention policy never enters the embedding store.
- An unreachable or slow embedding server must not break a session start: Task 2 tests the 1.5 s connect/read timeout, the `embedder-unavailable` fallback and unchanged lexical results.
- A vector cached under an old fingerprint must never be compared with a query vector from a new model: Task 2 tests fingerprint = model id + dimension + server-reported model.
- The MCP server runs inside a sandboxed agent: Task 4 tests that it never writes under `~/.claude` or the memory root, and that malformed JSON-RPC lines produce an error response, not a crash.
- A subset consolidation must not cause duplicate topic files: Task 6 tests that MEMORY.md is always in the prompt, that a `new` suggestion whose slug matches an existing file is rewritten to `update`, and that the weekly full audit still sees every file.
- A `patch` whose `find` text matches zero or several places refuses to apply: Task 6 tests both and the deferral path.
- Usage data must never delete anything on its own: Task 5 tests that it only reorders ties and annotates removal candidates.
- The private eval set must stay out of the repository: Task 1 tests the `.gitignore` entry and the refusal to calibrate on public fixtures.

## Shared interfaces

`recall_adapters.OpenAICompatibleEmbedder(base_url, model, dimension, timeout_seconds, api_key_env=None)` implements the existing `Embedder` protocol. `fingerprint` is `f"openai-compatible:{model}:{dimension}"`; `remote_data_egress` is `False` only when the URL host is loopback, otherwise it must equal the configured flag. `embed(texts)` posts to `/v1/embeddings` with `urllib.request`, validates the response shape and dimension, and raises `AdapterError` on any failure.

`recall_embeddings.vector_to_blob(values) -> bytes` and `blob_to_vector(blob) -> array.array` store L2-normalized float32 vectors; `recall_embeddings.dense_candidates(conn, query_vector, fingerprint, allowed_kinds, limit=200) -> list[tuple[str, float]]` scans the store in pure Python and returns `(document_id, cosine)` sorted descending.

`recall_query.rank_lexical(conn, query, *, dense=None)` accepts an optional pre-computed dense list and adds it as a fifth RRF list named `dense`; everything else is unchanged.

`recall_context.recall_for_agent(conn, settings, *, query_text, cwd, project_slug=None, kinds=None, limit=5, budget_codepoints=4000) -> RecallResult` is the single entry point used by `dream recall` and the MCP server; it builds a `RecallQuery` with `hook_event = "agent-query"` so selection applies the calibrated threshold.

`mcp_server.serve(stdin, stdout, config)` implements JSON-RPC 2.0 over newline-delimited stdio with `initialize`, `tools/list`, `tools/call`, `ping`; tools are `recall_search`, `recall_get`, `memory_index`.

`recall_usage.harvest(conn, marker_root) -> int` moves selected-document markers into `recall_usage(document_id, first_selected_at, last_selected_at, select_count)` and deletes harvested markers; `recall_usage.usage_map(conn) -> dict[str, tuple[int, str]]` serves ranking and the audit.

`consolidate.select_memory_subset(conn, memory_root, batch, settings, *, embedder=None) -> list[Path]` returns MEMORY.md plus at most `consolidate.subset_files` topic files ranked by the batch's findings; `consolidate.Suggestion` gains `find: str | None` for `kind = "patch"`; `dream.apply_patch(target, find, replace, *, expected_sha)` is the only writer for patches.

`providers.openai_compatible.generate(request, settings) -> GenerationResult` posts chat completions with a JSON schema response format to a configured `base_url`, records `provider = "openai-compatible"` and the server-reported model in provenance, and never retries a non-timeout error.

## Task order

### Task 0: Spec and baseline gate

**Steps:**

- [ ] Run `python3 -m pytest -q -p no:cacheprovider dream`; require the current 489 tests to pass.
- [ ] Run `git status --short` and record unrelated changes; create `feat/recall-upgrade` from `main`.
- [ ] Write `docs/superpowers/specs/2026-10-19-recall-and-consolidation-upgrade-design.md` from the review tab: objective, data model changes (`recall_embeddings` blob column, `recall_usage` table, `patch` suggestion kind), config keys with defaults, hook and MCP contracts, non-goals (no numpy, no sqlite-vec, no character changes).
- [ ] Stop and request Szymon's approval of the spec. Do not start Task 1 without it.

### Task 1: Private held-out recall eval with per-language metrics

This task comes first because it is the decision gate for every retrieval change that follows.

**Files:**

- Create `dream/eval_build.py`
- Modify `dream/recall_eval.py`
- Modify `dream/dream.py` (`eval-build` subcommand, `recall-eval --by-language`)
- Create `dream/test_eval_build.py`
- Modify `dream/test_recall_eval.py`
- Modify `.gitignore`

**Interfaces:**

```python
build_eval_fixture(conn, memory_root, *, config, questions_per_file=3, languages=("en", "pl"), limit=None) -> dict
evaluate_fixture_file(path, conn, settings=None, *, embedder=None) -> EvaluationReport  # gains per-language and per-mode breakdown
```

`build_eval_fixture` samples approved memory files and distilled summaries, asks the `distill` stage provider for `questions_per_file` natural questions per document in each language, and writes `~/.config/dream/eval/private.json` with the same `{documents, queries}` shape as the public fixture, each query tagged `language` and `expected_ids`. The model receives the retention policy context; excluded documents are skipped. The output path is outside the repository and `.gitignore` gains `dream/fixtures/private*`.

`recall-eval` reports Recall@1, Recall@5, MRR, forbidden-result count and p95 latency per language and per mode (`lexical`, `lexical_plus_embedder`). `recall-calibrate` refuses a fixture whose path is under `dream/fixtures/` with the existing public-fixture error.

Tests use a fake provider that returns fixed questions and assert fixture shape, language tags, policy exclusion, the gitignore entry and the per-language report. No real model calls.

Run:

```bash
python3 -m pytest -q -p no:cacheprovider dream/test_eval_build.py dream/test_recall_eval.py dream
git add dream/eval_build.py dream/recall_eval.py dream/dream.py dream/test_eval_build.py dream/test_recall_eval.py .gitignore
git commit -m "feat: build private per-language recall eval fixtures"
```

After the commit, Szymon runs `dream eval-build --limit 100` and `dream recall-eval --fixtures ~/.config/dream/eval/private.json --by-language` once with real providers and records the lexical baseline in the spec. The expected finding is near-zero Recall@5 for Polish queries; that number is the bar Task 2 must clear.

### Task 2: Local OpenAI-compatible embedder and dense candidates in RRF

**Files:**

- Modify `dream/recall_adapters.py` (`OpenAICompatibleEmbedder`)
- Create `dream/recall_embeddings.py`
- Modify `dream/recall_query.py` (fifth list `dense`)
- Modify `dream/recall_context.py` (dense list computed before `rank_lexical` when the embedder is enabled)
- Modify `dream/recall_documents.py` (embed changed documents during nightly synchronization)
- Modify `dream/config.py`, `dream/default-config.toml`, `dream/schema.sql` (migration adds `vector_blob BLOB`, `dimension INTEGER` to `recall_embeddings`)
- Modify `dream/dream.py` (`dream embed --refresh`, `preflight` reports embedder reachability)
- Create `dream/test_recall_embeddings.py`, `dream/test_recall_embedder_http.py`
- Modify `dream/test_recall_adapters.py`, `dream/test_recall_query.py`, `dream/test_recall_preflight.py`

**Configuration:**

```toml
[recall.embedder]
enabled = false
type = "openai-compatible"
base_url = "http://127.0.0.1:8081"
model = "bge-m3"
dimension = 1024
timeout_seconds = 1.5
remote_data_egress = false
dense_candidates = 200
```

`type = "none"` keeps today's behaviour. A non-loopback `base_url` with `remote_data_egress = false` is a configuration error that `preflight` and `context` report (context fails open to lexical).

**Behaviour:**

- Nightly synchronization embeds every `recall_documents` row whose `(content_sha256, fingerprint)` has no vector, in batches of 32, after retention filtering. Transcript documents are embedded only when `allow_raw_transcript_prompt = true`. A server failure leaves the lexical index complete and logs one diagnostic.
- Vectors are L2-normalized float32 blobs; `vector_json` stays for backward compatibility and is read when the blob is missing.
- At query time the embedder embeds `query.query_text` once (session start uses cwd and roots, as today), `dense_candidates` scans vectors of the current fingerprint in pure Python (`array` dot products over pre-normalized vectors) and returns the top `dense_candidates` ids. `rank_lexical` fuses them as a fifth list so a document reachable only by meaning still enters the result. The prompt-path rule that keeps only FTS matches is relaxed to FTS-or-dense matches.
- The existing `apply_optional_adapters` rerank over the top 100 stays, now reading blobs.
- Scrubber and renderer are unchanged and still run on every candidate.

**Model choice:** decided by Task 1's private fixture, not by opinion. Candidates to run through `llama-server --embedding` on the 3090s: `bge-m3` (1024-d, multilingual), `multilingual-e5-large` (1024-d), `Qwen3-Embedding-0.6B` (1024-d) and `Qwen3-Embedding-4B` (2560-d). Pick the smallest model whose Polish Recall@5 is within 3 points of the best; record the choice and numbers in the spec.

**Tests** use a loopback `http.server` fake that returns deterministic vectors keyed by text, plus failure modes (connection refused, timeout, wrong dimension, malformed JSON). They assert: fallback to lexical with `embedder-unavailable`; fingerprint mismatch never compares vectors; retention-excluded documents are never embedded; a Polish query reaches an English document through the dense list; p95 of the pure-Python scan over 5,000 synthetic 1024-d vectors stays under 400 ms on CI; dense-only candidates carry the marker and scrubbed excerpt.

Run:

```bash
python3 -m pytest -q -p no:cacheprovider dream/test_recall_embeddings.py dream/test_recall_embedder_http.py dream/test_recall_adapters.py dream/test_recall_query.py dream/test_recall_preflight.py dream
git add dream/recall_adapters.py dream/recall_embeddings.py dream/recall_query.py dream/recall_context.py dream/recall_documents.py dream/config.py dream/default-config.toml dream/schema.sql dream/dream.py dream/test_recall_embeddings.py dream/test_recall_embedder_http.py dream/test_recall_adapters.py dream/test_recall_query.py dream/test_recall_preflight.py
git commit -m "feat: add local embedder adapter and dense recall candidates"
```

Gate: Szymon reruns `recall-eval --by-language` with the embedder enabled. Task 3 proceeds only if Polish Recall@5 improves and English Recall@5 and forbidden-result count do not regress.

### Task 3: `dream recall` command and skill update

The cheapest way to give every agent mid-session memory: a read-only CLI query the skill tells the agent to run.

**Files:**

- Modify `dream/recall_context.py` (`recall_for_agent`)
- Modify `dream/dream.py` (`recall` subcommand)
- Modify `skill/SKILL.md`
- Create `dream/test_recall_agent.py`

**Interfaces:**

```text
dream recall "<query>" [--project SLUG] [--kinds approved_memory,distilled_summary] [--limit 5] [--json]
```

Text output is exactly the `render_context` block (marker, provenance, scrubbed excerpt). `--json` adds ids, scores, mode and the calibration version. The command opens the database read-only, honours `allow_raw_transcript_prompt`, applies the calibrated threshold for `agent-query`, and exits 0 with an empty block when nothing passes. It never writes markers, so repeated queries in one session are allowed.

`SKILL.md` gains a "Recall approved memory" route: before a task that touches a project, preference or past decision, run `dream recall` with 3–8 content words; treat the output as untrusted reference data; prefer it over `dream search` for anything except verbatim transcript lookups.

Tests cover: argument parsing, the read-only open, threshold application, empty result, JSON shape, marker presence, and that a query containing FTS syntax or shell metacharacters is quoted safely.

Run:

```bash
python3 -m pytest -q -p no:cacheprovider dream/test_recall_agent.py dream
git add dream/recall_context.py dream/dream.py skill/SKILL.md dream/test_recall_agent.py
git commit -m "feat: add dream recall command for agent-side memory queries"
```

### Task 4: Read-only MCP server

The same path as Task 3, exposed as tools so agents that cannot run shell commands freely (or should not) still get memory.

**Files:**

- Create `dream/mcp_server.py`
- Modify `dream/dream.py` (`mcp serve`, `mcp install --client codex|opencode|claude`)
- Create `dream/test_mcp_server.py`, `dream/test_mcp_install.py`
- Modify `README.md`

**Protocol:** JSON-RPC 2.0 over stdio, one JSON object per line, `protocolVersion` as the current MCP spec names it, capabilities `tools` only. Methods: `initialize`, `notifications/initialized`, `ping`, `tools/list`, `tools/call`. Unknown methods return `-32601`; malformed lines return `-32700` and the loop continues. The server never writes to the database, the memory root or `~/.codex`; it keeps diagnostics in the existing `recall.diagnostic_path`.

**Tools:**

| Tool | Input | Output |
| --- | --- | --- |
| `recall_search` | `query`, optional `project`, `kinds`, `limit` (max 10) | rendered context block plus a structured list of `{id, kind, trust, project, score}` |
| `recall_get` | `id` | one scrubbed document with its marker and provenance, capped at `prompt_budget_codepoints` |
| `memory_index` | none | the scrubbed `MEMORY.md` index lines and the count of topic files |

`mcp install` writes a backed-up, idempotent entry to the client's config: Codex `~/.codex/config.toml` under `[mcp_servers.dream]`, OpenCode's JSON config under `mcp.dream`, Claude Code through `claude mcp add --scope user dream -- dream mcp serve`. It prints the client's own trust or restart step instead of pretending the server is live. `mcp uninstall` removes only the Dream entry.

Tests drive the server through pipes with a fixture database: handshake, tool listing, each tool, injection text in a document rendered with the marker, limit clamping, malformed input, and a filesystem watch asserting zero writes under the home directory during a session. Install tests patch a temporary home and assert idempotence and preserved unrelated entries.

Run:

```bash
python3 -m pytest -q -p no:cacheprovider dream/test_mcp_server.py dream/test_mcp_install.py dream
git add dream/mcp_server.py dream/dream.py dream/test_mcp_server.py dream/test_mcp_install.py README.md
git commit -m "feat: expose read-only recall through a stdio MCP server"
```

### Task 5: Usage harvesting and usage-aware ranking and decay reporting

Hooks are read-only, so usage must be collected by the nightly write path from the markers hooks already leave under `/tmp/dream-recall-<uid>/` and from `recall_events.selected_ids_json`.

**Files:**

- Create `dream/recall_usage.py`
- Modify `dream/schema.sql` (migration adds `recall_usage`)
- Modify `dream/ingest.py` (harvest at the start of `dream ingest`)
- Modify `dream/recall_query.py` (usage as the tie-break after trust, before best component rank)
- Modify `dream/consolidate.py` (usage annotations in the staleness audit input)
- Modify `dream/dream.py` (`dream status --usage`)
- Modify `dream/metrics.py` (`dream_recall_documents_unused_90d` gauge)
- Create `dream/test_recall_usage.py`
- Modify `dream/test_recall_query.py`, `dream/test_consolidate_prompt.py`, `dream/test_metrics.py`

**Behaviour:**

- `harvest` reads each marker with `status = succeeded`, upserts one `recall_usage` row per selected document, deletes the marker, and ignores markers younger than 10 s (possibly still being written). Markers from `dream recall` and the MCP server are not written (Task 3), so interactive queries do not inflate usage; a later task may add an opt-in.
- Ranking: usage never adds or removes candidates. It is a tie-break only, and only when two candidates have equal RRF score and trust.
- Audit input: each memory file in the consolidation prompt carries `last recalled: <ISO date or never>; recalls: <n>` so the model can cite evidence, mirroring the Codex native design. The model is still forbidden from using age alone; `_drop_fresh_removals` is unchanged.
- `dream status --usage` lists approved files never recalled in 90 days with their index line, for human review. Nothing is deleted.
- The exporter gains one numeric gauge; it emits no document ids.

Tests use fixed clocks and synthetic markers: harvest idempotence, the 10 s guard, tie-break-only ranking (a high-usage irrelevant document never enters results), prompt annotations, the status report, and the gauge.

Run:

```bash
python3 -m pytest -q -p no:cacheprovider dream/test_recall_usage.py dream/test_recall_query.py dream/test_consolidate_prompt.py dream/test_metrics.py dream
git add dream/recall_usage.py dream/schema.sql dream/ingest.py dream/recall_query.py dream/consolidate.py dream/dream.py dream/metrics.py dream/test_recall_usage.py dream/test_recall_query.py dream/test_consolidate_prompt.py dream/test_metrics.py
git commit -m "feat: harvest recall usage for ranking ties and the staleness audit"
```

### Task 6: Incremental consolidation, `patch` suggestions, weekly audit stage

Splits today's single 800k-character call into a nightly subset merge and a weekly full audit, and gives the model a partial-edit primitive so review diffs shrink.

**Files:**

- Modify `dream/consolidate.py` (`select_memory_subset`, `patch` kind, `mode` handling)
- Create `dream/prompts/consolidate-subset.md`, `dream/prompts/audit.md`
- Modify `dream/prompts/consolidate.md` (adds the `patch` kind; otherwise unchanged)
- Modify `dream/dream.py` (`apply_patch`, `consolidate --mode full|subset`, `audit` subcommand, review and curate paths for `patch`)
- Modify `dream/curate.py` (curation schema accepts `patch`)
- Modify `dream/config.py`, `dream/default-config.toml` (`[consolidate] mode = "full"`, `subset_files = 15`; `[stages.audit]`)
- Modify `dream/systemd/dream.service` (nightly `consolidate`), create `dream/systemd/dream-audit.service` and `dream-audit.timer` (weekly)
- Modify `dream/schema.sql` (suggestions gain `find_text TEXT`)
- Create `dream/test_consolidate_subset.py`, `dream/test_patch_apply.py`, `dream/test_audit.py`
- Modify `dream/test_consolidate_memory.py`, `dream/test_curate.py`, `dream/test_consolidate_removal_gate.py`

**Subset selection:** build one query per batch from the distilled findings (all `preferences`, `projects`, `decisions` and `user_facts` strings joined), run it through `rank_lexical` restricted to `approved_memory` documents (with the dense list when the embedder is enabled), and take the top `subset_files` files plus MEMORY.md, plus every file whose slug appears verbatim in a finding. The prompt states that the store is larger than what is shown and lists the remaining index lines so the model proposes `update` rather than `new` for a topic that exists. A `new` suggestion whose `target_path` already exists is converted host-side to `update` with the current body as base.

**Patch kind:** `{kind: "patch", target_path, find, body (replacement), rationale, source_sessions}`. `apply_patch` requires the SHA precondition, exactly one occurrence of `find` after NFC normalization, and a non-empty replacement unless the rationale states a deletion; zero or multiple matches defer the suggestion with a reason. Preview files under `.suggestions/` show a unified diff. `accept`, `merge`, `curate-configured` and `review` all handle `patch`; `index` lines are still appended, never patched.

**Weekly audit:** `dream audit` runs the staleness pass over the full store in chunks of at most `MAX_CURRENT_MEMORY_CHARS / 4`, carries usage annotations from Task 5, and emits only `remove` and `patch` suggestions. `_drop_fresh_removals` and the 21-day grace apply unchanged. The nightly service no longer performs the audit when `mode = "subset"`.

**Mode switch:** `mode = "full"` keeps today's behaviour byte-for-byte so the switch is reversible. Szymon flips to `subset` after one week of comparing accepted-suggestion counts and consolidation duration from `run.json`.

Tests use fake providers and a temporary memory root of 60 files: the subset always contains MEMORY.md; slug-mentioned files are included; `new` on an existing path becomes `update`; patch application, ambiguity, deferral, backups and SHA refusal; audit chunking covers every file exactly once; the systemd units contain no provider commands; `mode = "full"` produces the identical prompt to today's.

Run:

```bash
python3 -m pytest -q -p no:cacheprovider dream/test_consolidate_subset.py dream/test_patch_apply.py dream/test_audit.py dream/test_consolidate_memory.py dream/test_curate.py dream/test_consolidate_removal_gate.py dream
git add dream/consolidate.py dream/prompts dream/dream.py dream/curate.py dream/config.py dream/default-config.toml dream/systemd dream/schema.sql dream/test_consolidate_subset.py dream/test_patch_apply.py dream/test_audit.py dream/test_consolidate_memory.py dream/test_curate.py dream/test_consolidate_removal_gate.py
git commit -m "feat: subset consolidation with patch suggestions and a weekly audit"
```

### Task 7: Local OpenAI-compatible generation provider

Lets `distill` (low reasoning, one call per session) run on the local Qwen executor instead of the ChatGPT subscription, and makes the embedder's server one of two uses of the same loopback endpoint pattern. Independent of Tasks 1–6.

**Files:**

- Create `dream/providers/openai_compatible.py`
- Modify `dream/providers/registry.py` (`type = "openai-compatible"`)
- Modify `dream/config.py`, `dream/default-config.toml` (a commented example provider), `dream/dream.py` (`preflight` checks reachability and `/v1/models`)
- Create `dream/test_provider_openai_compatible.py`
- Modify `dream/test_provider_architecture.py`, `dream/test_optional_model.py`

**Configuration:**

```toml
[providers.local-qwen]
type = "openai-compatible"
base_url = "http://127.0.0.1:8080"
model = "qwen3-32b"
api_key_env = "LOCAL_LLM_API_KEY"
remote_data_egress = false
timeout_seconds = 600
```

`generate` sends `/v1/chat/completions` with the stage's system prompt, the user prompt and `response_format = {type: "json_schema", json_schema: {...}}`; when the server rejects `json_schema` it retries once with `{type: "json_object"}` and validates against the stage schema host-side, as `backend.validate_output` already does for CLI providers. Usage fields map onto `input_tokens` and `output_tokens`. A non-loopback `base_url` with `remote_data_egress = false` is a configuration error. No streaming, no tool calls.

Tests use the loopback fake server: schema-strict success, fallback to `json_object`, invalid JSON rejected, timeout surfaced as a provider failure that leaves no distillation row, and provenance recording the server-reported model.

Run:

```bash
python3 -m pytest -q -p no:cacheprovider dream/test_provider_openai_compatible.py dream/test_provider_architecture.py dream/test_optional_model.py dream
git add dream/providers/openai_compatible.py dream/providers/registry.py dream/config.py dream/default-config.toml dream/dream.py dream/test_provider_openai_compatible.py dream/test_provider_architecture.py dream/test_optional_model.py
git commit -m "feat: add OpenAI-compatible provider for local model servers"
```

After the commit, Szymon routes `[stages.distill]` to `local-qwen` on 20 sessions and compares distilled notes against the Codex baseline before switching the nightly stage.

### Task 8: Lexical path performance, documentation, final gate

**Files:**

- Modify `dream/recall_query.py` (structured and path matching only over FTS-or-dense candidates plus the 500 most recent documents; scrub once per candidate, not per list)
- Modify `dream/recall_documents.py` (store a pre-normalized `path_text` column for path matching; migration)
- Modify `README.md`, `skill/SKILL.md`, `docs/how-dream-works.html` (new stages and commands)
- Modify `dream/test_recall_query.py` (p95 budget test on 10,000 synthetic documents)

The session-start path must stay within the spec's 150 ms p95 and the prompt path within 250 ms on the 10,000-document fixture, measured in the test with a warmed connection. Results must be identical to the current implementation on the existing fixtures (a golden-output test guards this).

Final gate:

- [ ] `python3 -m pytest -q -p no:cacheprovider dream` passes; the count exceeds 489.
- [ ] `git diff --check` is clean.
- [ ] `dream preflight` on a fresh config reports embedder disabled, MCP not installed, consolidation mode `full`, and no errors.
- [ ] README documents every new config key with its default.
- [ ] Open a PR from `feat/recall-upgrade` to `main` with the spec, the eval numbers Szymon recorded, and the per-task commit list. Do not merge.

Run:

```bash
python3 -m pytest -q -p no:cacheprovider dream
git add dream/recall_query.py dream/recall_documents.py README.md skill/SKILL.md docs/how-dream-works.html dream/test_recall_query.py
git commit -m "perf: bound lexical recall work and document the recall upgrade"
```

## Dependencies and parallelism

Task 0 gates everything; Task 1 gates Task 2; Task 2's eval gate decides whether the dense path ships. The rest fans out.

- Serial: Task 0 → Task 1 → Task 2 (with Szymon's eval run between 1 and 2, and after 2).
- Parallel after Task 2, each in its own worktree or subagent: Task 3, Task 5, Task 7. Task 4 starts after Task 3 lands. Task 6 can start after Task 2 but should land after Task 5 so the audit gets usage annotations.
- Task 8 is last and runs on the merged branch.
- Three human checkpoints, none skippable: spec approval (Task 0), lexical baseline and embedder choice (Tasks 1–2), the `mode = "subset"` switch after a week of comparison (Task 6).
- If any task fails its gate, the branch still ships the tasks before it; nothing here is all-or-nothing.

## Acceptance checklist

- [ ] Spec approved before code; every new config key defaults to current behaviour.
- [ ] Private eval fixture lives outside the repository; `recall-calibrate` refuses public fixtures; per-language, per-mode report exists.
- [ ] Embedder adapter runs against a loopback OpenAI-compatible server through `urllib`; fallback to lexical on any failure; fingerprints bind vectors to model and dimension; retention-excluded documents are never embedded.
- [ ] Dense candidates enter RRF as a fifth list; a Polish query reaches an English approved memory in tests; scrubber and marker apply to every candidate.
- [ ] `dream recall` and the MCP server are read-only, render through `render_context`, and never write under the home directory; install and uninstall are idempotent and preserve unrelated entries.
- [ ] Usage is harvested from markers, reorders ties only, annotates the audit, and deletes nothing.
- [ ] `mode = "full"` is byte-identical to today's consolidation; `subset` always includes MEMORY.md and converts `new`-on-existing to `update`; `patch` applies under SHA preconditions with one exact match or defers; the weekly audit covers every file once.
- [ ] OpenAI-compatible generation provider validates schema output host-side and records provenance; non-loopback URLs require explicit egress consent.
- [ ] Session-start p95 ≤ 150 ms and prompt p95 ≤ 250 ms on the 10,000-document fixture; golden outputs unchanged.
- [ ] Character pipeline untouched; full suite passes; PR opened, not merged.
