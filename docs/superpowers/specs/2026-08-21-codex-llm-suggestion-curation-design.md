# Codex LLM suggestion curation design

## Status

Proposed design for review before implementation.

## Context

Dream already uses the configured `codex` provider for distillation and
consolidation. Consolidation creates pending suggestions, but the nightly
`suggestions apply-configured` command applies them mechanically. It does not
ask a model to decide whether a suggestion should be accepted, rejected, or
merged. A changed target therefore remains pending as a conflict instead of
being resolved with the current file as context.

The earlier Claude Code workflow supplied this judgment step in-session. The
nightly workflow must provide the same judgment through Codex CLI, using the
configured ChatGPT-subscription provider and never the Claude provider.

## Goal

Add a provider-neutral LLM curation stage whose configured production route is
`codex`, and make the nightly timer use it before writing memory files.

The stage must let the model choose one of:

- `accept`: the suggestion is current and can be applied;
- `reject`: the suggestion is stale, redundant, or unsupported;
- `merge`: write a model-produced merged body that preserves newer live facts;
- `defer`: leave the suggestion pending when the evidence is insufficient.

The host, not the model, remains authoritative for target paths, file writes,
backups, SHA preconditions, protected files, and database status changes.

## Configuration and routing

Add a `review` model stage to the existing stage configuration, with the local
profile explicitly routed through the `codex` provider using model
`gpt-5.6-luna` and `reasoning_effort = "high"`. The systemd unit must invoke
the LLM curation command; it must not invoke Claude or the deterministic
accept-all path.

The deterministic apply command remains available for explicit emergency use,
but it is no longer the nightly path.

## Data flow

1. Load all pending suggestions and their current target contents.
2. For each suggestion, provide the model with the proposed body, rationale,
   source sessions, current body, base SHA, target kind, and conflict status.
3. Ask the configured Codex provider for strict JSON containing only the
   decision, optional replacement/merged body, and a concise reason.
4. Validate the decision schema and bind it to the database suggestion ID;
   the model cannot choose or invent a target path.
5. Before applying each non-deferred decision, reread the target and compare
   its SHA with the SHA supplied to the model. A changed target is deferred,
   not overwritten.
6. Create one backup before the first write and reuse the existing guarded
   apply path for accepted or merged decisions. Rejected and deferred rows do
   not write memory files.
7. Record the decision and reason in the run output; keep deferred rows
   pending for a later run.

For `MEMORY.md` index suggestions, the model returns only intended index lines
and the host retains the existing append-only merge behavior. For regular
memory files, a merge returns the complete replacement body.

## Failure and safety behavior

- A provider, timeout, or schema error aborts the curation run before any
  decision from that run is written.
- A concurrent file change defers only the affected suggestion and never
  bypasses the SHA precondition.
- A path or protected-file violation is rejected by the existing host-side
  checks, regardless of model output.
- The model cannot mark a row accepted without the guarded host apply
  succeeding.
- Existing memory backups remain the rollback mechanism, with the configured
  retention policy.
- The report distinguishes model decisions, applied writes, rejects, deferrals,
  conflicts, and provider failures.

## Compatibility

Existing `suggestions list`, `accept`, `reject`, `merge`, and deterministic
`apply-configured` commands retain their current contracts. The new curation
command is the only change to the nightly service behavior. No native Codex
Memories state or automatic recall hook is changed by this feature.

## Testing and acceptance

- Unit tests cover the decision schema, Codex stage routing, all four decisions,
  index-specific body handling, provider failure before writes, and concurrent
  SHA changes.
- Integration tests use a fake structured Codex response and verify database
  statuses, file contents, previews, and backup creation.
- A provider-import test verifies the curation path loads `codex-cli` without
  importing the Claude adapter.
- The full test suite, `git diff --check`, and a dry-run against the current
  pending suggestions must pass.
- The live nightly unit must show the Codex curation command in its exact
  deployed command chain and no Claude command.
