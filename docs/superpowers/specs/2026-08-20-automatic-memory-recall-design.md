# Automatic Memory Recall Design

**Status:** approved in conversation on 2026-08-20  
**Target:** post-`v0.0.1` feature development  
**Scope:** Codex integration first, with provider-neutral retrieval interfaces

## Objective

Dream shall supply useful memories automatically when a Codex session starts and when its first user prompt arrives. The public repository must work without an embedding model, reranker, hosted account, or machine-specific service. Optional embedders and rerankers may improve recall quality but must never be required for a session to start.

The retrieval domain is global. A current working directory, repository, or project slug contributes ranking evidence but never excludes memories from other projects, because one session may span several repositories and operational areas.

## Release boundary

The repository's current behavior is the initial `v0.0.1` release. Automatic recall is developed after that baseline and must not be included retroactively in the `v0.0.1` tag.

## User-visible behavior

### Session start

A Codex `SessionStart` hook invokes `dream context session-start`. Dream returns a concise global index built from approved memory files and recent distilled session summaries. The output is valid additional developer context and contains source labels sufficient for later inspection.

The startup context is bounded by a configurable budget. It favors approved memory index entries, recent summaries, and entries related to the current directory, but reserves space for cross-project entries. Startup never performs model inference.

### First prompt

A Codex `UserPromptSubmit` hook invokes `dream context prompt` with hook JSON on standard input. The command extracts the prompt, runs baseline lexical retrieval, and returns only results above a configured confidence threshold. Results are concise memory excerpts with provenance, injected as additional developer context.

The prompt path runs once per new session by default. Repeated events for the same session return no additional context unless configuration explicitly enables per-prompt recall, preventing duplicate context and recurring latency.

### Graceful degradation

Recall uses the strongest configured path that is available:

1. FTS5 candidate retrieval, optional semantic candidates, and optional reranking.
2. FTS5 candidate retrieval and optional semantic candidates when reranking is unavailable.
3. FTS5 retrieval when both optional model layers are unavailable.
4. Session-start index only when prompt retrieval fails.

Hook commands fail open: they record a diagnostic and return success without additional context when retrieval cannot complete. They never prevent Codex from starting or accepting a prompt.

## Retrieval architecture

### Canonical types

`RecallQuery` contains the prompt, session identifier, current working directory, detected repository roots, and budget. `RecallCandidate` contains stable source identity, text, source kind, timestamps, project and path metadata, component scores, and provenance. `RecallResult` contains selected candidates, rendered context, diagnostics, and the retrieval mode used.

### Baseline candidate generation

Baseline retrieval requires only Python and the existing SQLite database. It combines:

- FTS5 matches against approved memory content, distilled summaries, and optionally raw transcript messages;
- exact and normalized matches for repository names, paths, ticket identifiers, commands, technologies, and named entities recoverable without a model;
- recency scoring;
- a soft current-project or current-directory boost;
- source-quality weighting, with approved memories above distilled summaries and raw transcripts;
- reciprocal-rank fusion, which combines independently ranked lists without requiring comparable raw scores.

Project metadata is never a hard filter. Candidate generation includes a globally ranked allocation so a prompt mentioning several systems can retrieve evidence from each.

### Optional embedder

An `Embedder` interface accepts texts and returns numeric vectors plus an implementation fingerprint. When configured and available, it contributes a semantic candidate list fused with baseline lists. Embeddings are cached in SQLite and invalidated when source content or the embedder fingerprint changes.

The core package has no mandatory model dependency. An optional installation extra supplies a recommended local multilingual embedding implementation and may download its model on first use. Other adapters may target Ollama or OpenAI-compatible endpoints without changing the pipeline.

### Optional reranker

A `Reranker` interface accepts a query and bounded candidate list and returns relevance scores. It changes ordering and may reject candidates below its threshold, but does not own candidate discovery. Failure is isolated and causes deterministic fallback to fused baseline ordering.

The optional local implementation may download a multilingual cross-encoder model on first use. Remote and user-provided adapters remain separate optional integrations.

### Selection and rendering

Selection removes duplicate excerpts, groups overlapping candidates from the same source, limits dominance by one session, and preserves diversity across repositories and source kinds. It stops at the context budget and adds nothing when all candidates remain below the applicable confidence threshold.

Every rendered item includes a stable source identifier and short origin label. Model scores, private home-directory prefixes, and database locations remain in diagnostics rather than agent context.

## Data scope and privacy

Approved Markdown memories and distilled summaries are searched by default. Raw transcript search is disabled for automatic injection by default and can be enabled explicitly because transcripts contain more noise and may contain sensitive data.

All recall indexes remain local unless a user explicitly configures a remote embedder or reranker. Configuration must state when content leaves the machine. Hook output must not expose absolute home-directory prefixes, detected secrets, or raw configuration values.

## Configuration

The configuration gains a `[recall]` section with explicit defaults for enablement, data sources, budgets, thresholds, first-prompt-only behavior, and failure logging. Nested `[recall.embedder]` and `[recall.reranker]` sections are disabled by default and select implementations by registered name.

Invalid optional-model configuration is reported by `dream preflight`. Runtime degrades to the next available tier unless strict mode is explicitly enabled for testing or diagnostics.

## Codex installation

`install.sh` continues installing the CLI, skill, and default configuration, and additionally offers an idempotent hook installation command instead of overwriting user hook files silently. Hook configuration is merged while preserving unrelated user and plugin hooks. Uninstall support removes only entries owned by Dream.

Generated hook definitions invoke the installed `dream` executable with conservative timeouts and use the official `SessionStart` and `UserPromptSubmit` output shape with `hookSpecificOutput.additionalContext`.

## Observability

`dream context ... --explain` emits machine-readable diagnostics outside injected context: candidate sources, component scores, fusion rank, model availability, fallback reason, selected character count, and elapsed time. Normal hooks are silent except for bounded local diagnostic logs.

The database records one recall event per hook invocation and selected candidate identities. It does not duplicate prompt bodies because prompts already exist in ingested transcripts.

## Quality evaluation

A public fixture suite represents single-project, cross-project, multilingual, ambiguous, stale-memory, duplicate, private-data, and no-relevant-memory cases. Each fixture declares relevant and forbidden source identifiers.

The evaluation command reports Recall@k, mean reciprocal rank, precision of injected results, forbidden-result count, empty-result correctness, context size, and latency. The baseline FTS5 path establishes the compatibility floor; optional model configurations must improve the same fixtures without increasing forbidden-result count.

## Error handling

- Missing database or tables: return no prompt context and preserve the startup index if it can be built from files.
- Invalid FTS syntax derived from natural language: tokenize and quote safe terms instead of passing the prompt directly to `MATCH`.
- Missing or failed optional model: log the adapter failure and continue with remaining candidate lists.
- Hook input without a prompt or session identifier: return success with no additional context.
- Oversized content: truncate only at excerpt boundaries and remain within budget.
- Concurrent ingestion and recall: use short read transactions, SQLite busy timeout, and no schema mutation during hooks.

## Compatibility and non-goals

The first integration target is Codex. Provider-neutral retrieval and rendering interfaces are required, but Claude Code hook installation is outside this feature. Dream does not become a background network service, require a vector database, automatically edit approved memories, or send local content to a remote model by default.

## Acceptance criteria

- A clean installation without optional extras injects a bounded session-start index and performs FTS5 recall on the first prompt.
- One prompt referring to two repositories can retrieve relevant memories from both, regardless of current working directory.
- Disabling or breaking the embedder and reranker does not block Codex and produces deterministic fallback output.
- Raw transcripts are absent from automatic recall unless explicitly enabled.
- Repeated installation does not add duplicate hooks.
- Injected context respects its budget, contains provenance, and excludes private absolute path prefixes.
- Evaluation compares baseline, embedder, reranker, and combined modes on identical fixtures.
- Existing ingest, distill, consolidate, review, search, status, and preflight tests remain passing.
