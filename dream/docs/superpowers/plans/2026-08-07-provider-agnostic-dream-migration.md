# Provider-Agnostic Dream Migration Plan

**Status:** implemented and validated on 2026-08-08

**Goal:** make `dream` independent of any transcript producer and LLM provider. The current installation must run entirely through the Codex CLI on the user's ChatGPT subscription, while Claude CLI, local models, or API-backed providers remain optional adapters selected by configuration.

**Current deployment profile:** Codex JSONL transcripts + `codex-cli` for both model stages. Existing memory and SQLite paths remain unchanged during migration.

## Invariants

- The core must not import provider-specific modules, use provider-specific model aliases, or know authentication details.
- Transcript source, model provider, storage, and user-facing skill are independent concerns.
- Missing optional providers must not break `status`, `ingest`, `search`, or another configured provider.
- No historical session, distillation, or suggestion may be lost during schema migration.
- A failed, truncated, or invalid model response must not advance consolidation state.
- Applying a suggestion must not overwrite a memory file changed since the suggestion was created.
- Existing nightly auto-apply behavior is preserved for this installation, but becomes an explicit configuration choice rather than a core assumption.
- All live-memory writes take a backup first.

## Target architecture

```text
Codex skill / Claude command / shell / systemd
                      |
                 neutral CLI
                      |
     +----------------+----------------+
     |                |                |
 transcript       stage router      storage/review
 sources              |                |
     |          model providers          |
 codex-jsonl     codex-cli            SQLite
 claude-jsonl    claude-cli           memory files
 future          local-cli / API      backups
```

The CLI orchestrates stages. A transcript source produces neutral session records. The stage router resolves `distill` or `consolidate` to a configured provider and model. Providers only execute fully resolved generation requests. Storage tracks source revisions, distillation provenance, consolidation consumption, suggestions, and apply preconditions.

## Configuration

Add a strict TOML configuration, defaulting to `~/.config/dream/config.toml`.

```toml
[storage]
db_path = "~/.claude/dream.db"
memory_root = "~/.claude/projects/-home-szymon/memory"

[stages.distill]
provider = "codex"
model = "gpt-5.6-luna"
reasoning_effort = "low"
timeout_seconds = 600

[stages.consolidate]
provider = "codex"
model = "gpt-5.6-sol"
reasoning_effort = "high"
timeout_seconds = 1800

[providers.codex]
type = "codex-cli"
auth = "chatgpt-subscription"

[providers.claude]
type = "claude-cli"

[[sources]]
name = "codex"
type = "codex-jsonl"
root = "~/.codex/sessions"
enabled = true

[[sources]]
name = "claude-history"
type = "claude-jsonl"
root = "~/.claude/projects"
enabled = false

[review]
mode = "auto-apply"
backup_keep = 14
```

Model identifiers belong only to deployment configuration. They are examples for the current profile, not built-in semantic aliases.

Configuration precedence, from strongest to weakest:

1. CLI flags.
2. Stage-specific environment variables.
3. TOML configuration.
4. Built-in safe defaults.

Reject unknown provider types, unknown stage options, missing provider references, and incompatible options at startup. Do not silently fall back to another provider.

Temporary compatibility:

- Map `DREAM_BACKEND` to both stages only when stage-specific configuration is absent.
- Accept old `--model haiku|opus` arguments only through a documented compatibility mapping and emit a deprecation warning.
- Preserve `CLAUDE_DREAM_DB` and `DREAM_MEMORY_ROOT` as legacy environment aliases.

## Neutral provider contract

Move shared types and errors out of `claude_cli.py` into a neutral module.

```python
@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    prompt: str
    schema: dict
    model: str
    timeout_seconds: int
    options: dict[str, object]


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_tokens: int | None
    cache_read_tokens: int | None


@dataclass(frozen=True)
class GenerationResult:
    output: dict
    provider: str
    model: str
    usage: Usage | None
    duration_ms: int


class ModelProvider(Protocol):
    def generate_structured(self, request: GenerationRequest) -> GenerationResult: ...
```

The stage router, not the provider, translates `distill` and `consolidate` into a configured provider, model, effort, timeout, and provider-specific options. Providers must not receive semantic stage names.

Provider rules:

- Register providers by configured type.
- Import an adapter only after it is selected.
- Validate structured output locally against the supplied schema before returning it.
- Return `usage=None` when a CLI does not expose usage; never invent zero-token usage.
- Normalize timeouts, authentication failures, invalid JSON, schema failures, and unavailable binaries into neutral exceptions.
- Keep subscription authentication and API-key policy inside the adapter/configuration, not the core.

Initial adapters:

```text
dream/providers/base.py
dream/providers/registry.py
dream/providers/codex_cli.py
dream/providers/claude_cli.py
```

`codex-cli` is the only required working model adapter for the current deployment. `claude-cli` remains optional and must not be imported or probed unless selected.

## Neutral transcript-source contract

Add a source interface separate from model providers.

```python
@dataclass(frozen=True)
class TranscriptRef:
    source: str
    external_session_id: str
    path: Path
    revision: str


@dataclass(frozen=True)
class ParsedSession:
    source: str
    external_session_id: str
    revision: str
    parser_version: str
    started_at: str | None
    ended_at: str | None
    cwd: str | None
    git_branch: str | None
    messages: list[Message]


class TranscriptSource(Protocol):
    def discover(self) -> Iterable[TranscriptRef]: ...
    def parse(self, ref: TranscriptRef) -> ParsedSession: ...
```

Use a stable content fingerprint as `revision`; mtime alone is insufficient. Resume or append creates a new revision and invalidates the previous distillation deliberately.

Initial sources:

```text
dream/sources/base.py
dream/sources/codex_jsonl.py
dream/sources/claude_jsonl.py
```

### Codex JSONL filtering

Treat the first `session_meta` row as file identity and metadata. Use both fields defensively:

- Accept normal user conversations where `payload.source == "cli"` and, when present, `payload.thread_source == "user"`.
- Reject `payload.source` objects containing `subagent`, including guardian and spawned-agent sessions.
- Reject `payload.thread_source == "subagent"` when present.
- Keep user messages only from `response_item/message`, `role=user`, `input_text` blocks.
- Filter every `input_text` block independently. Drop injected blocks beginning with `# AGENTS.md instructions for` or `<environment_context>` without dropping a real user block beside them.
- Keep assistant messages only from `response_item/message`, `role=assistant`, `phase=final_answer`, `output_text` blocks.
- Drop developer messages, commentary, reasoning, events, tool calls, and tool outputs.

Cover fork, resume, compaction, imported history, repeated final answers, guardian sessions, and spawned subagents with fixtures. Do not assume path layout alone identifies a user conversation.

## Database migration and provenance

Replace timestamp-only progress with explicit identities and revisions.

### Sessions

Add or migrate to fields equivalent to:

```text
internal_session_id
source
external_session_id
source_revision
parser_version
jsonl_path
timestamps/cwd/branch/counts
UNIQUE(source, external_session_id)
```

For existing rows:

- Backfill `source = "claude"`.
- Preserve the existing session identifier as `external_session_id`.
- Assign a deterministic internal identifier.
- Preserve messages and FTS row consistency.

Replace `INSERT OR REPLACE` with explicit UPSERT logic. When a source revision changes, replace messages intentionally and mark the prior distillation stale without relying on foreign-key cascade side effects.

### Distillations

Each distillation version must record:

```text
distillation_id
internal_session_id
source_revision
parser_version
prompt_version
provider
model
provider_options fingerprint
notes/summary/usage/duration
created_at
```

Changing the transcript revision, parser, prompt, provider, model, or material provider options produces a new distillation identity. Prompt version may be a content hash.

### Consolidation ledger

Deprecate `meta.last_consolidate_at` as the source of truth. Add a ledger that references the exact distillation versions successfully consumed by a consolidation run.

Required behavior:

1. Select unconsolidated distillation identities, independent of session timestamps.
2. Build batches within a measured serialized-character/token budget.
3. Never truncate a selected batch silently.
4. Persist suggestions and ledger entries in one transaction after a valid provider result.
5. Mark only the distillations actually included in that provider request.
6. On timeout, invalid JSON, schema failure, or persistence failure, consume nothing.
7. Historical Codex sessions imported after the old timestamp cursor must still be processed exactly once.

Staleness audit runs without new distillations remain allowed and do not create false ledger entries.

### Versioned migration

- Add a schema-version table and ordered, transactional migrations.
- Back up the SQLite database, including WAL/SHM state safely, before the first migration.
- Test migration from a realistic copy of the current database.
- Make every migration idempotent or explicitly version-gated.
- Preserve existing suggestions and their statuses.

## Suggestion safety and review API

Store the state on which each suggestion was based:

```text
base_sha256
target_existed
memory_snapshot/version
```

For `update`, `remove`, and `index`, refuse apply when the current file hash differs from `base_sha256`. For `new`, refuse apply when a target that was absent now exists. A conflict stays pending until explicitly merged or rejected.

Expose a machine-readable, granular CLI instead of having skills edit SQLite directly:

```text
dream suggestions list --json
dream suggestions accept <id>
dream suggestions reject <id>
dream suggestions merge <id> --body-file <path>
dream suggestions apply-all
```

Rules:

- Back up once before the first write in a review batch.
- Keep target-containment and protected-file checks.
- Validate expected hashes inside the same operation that writes the file.
- Remove preview files only after the DB status commits successfully.
- `merge` uses the current file as its base and records the resolved content explicitly.

Review mode is configuration, not provider behavior:

- `suggest-only`: nightly pipeline stops after producing proposals.
- `auto-apply`: preserve the current installation's mechanical nightly apply with backups and hash preconditions.
- Interactive `przejrzyj dream`: the active agent evaluates every proposal and calls granular CLI actions.

For new installations, default to `suggest-only`. During migration of this host, write `auto-apply` explicitly so behavior does not change accidentally.

## Codex skill

Create a thin skill at `skills/dream/` and install/link it into `~/.codex/skills/dream`.

```text
skills/dream/
  SKILL.md
  agents/openai.yaml
```

The skill must:

- Contain only `name` and `description` in `SKILL.md` frontmatter.
- Route `$dream`, status, ingest, distill, consolidate, search, and review requests to the neutral CLI.
- Read resolved configuration/status rather than encode provider or model names.
- Use the JSON suggestion API for autonomous accept/reject/merge decisions.
- Never modify the DB directly.
- Never imply that Codex is required by the core.
- Report adapter/authentication errors exactly and only for the selected provider.

Provider-specific operational details remain in adapter documentation or configuration examples, not in the skill body.

Validate with `quick_validate.py` and representative activation/non-activation prompts.

## Systemd and installation

Keep systemd provider-neutral:

```text
dream ingest
dream distill --yes --limit 50
dream consolidate
dream suggestions apply-all   # only when review.mode=auto-apply
```

The unit loads the normal dream configuration; it must not hard-code `DREAM_BACKEND=codex`. The current TOML profile selects `codex-cli`.

Installation changes:

- Install the neutral `dream` CLI.
- Install/link the Codex skill independently.
- Do not require or probe Claude CLI during a Codex-only install.
- Add an explicit provider preflight command that checks only configured providers without exposing credentials.
- Keep legacy Claude commands available when their files exist, but do not make them a runtime dependency.

## Implementation sequence

### 1. Characterize and protect the current state

- Add tests that freeze current Claude ingest, distill, consolidate, review, FTS, and nightly behavior.
- Capture anonymized Codex JSONL fixtures for main, subagent, guardian, resume, fork, and compaction cases.
- Create verified backups of the DB and memory tree before schema work.

### 2. Introduce neutral configuration and provider types

- Add strict TOML loading and precedence tests.
- Extract shared results, usage, errors, schema validation, and process helpers.
- Add the provider registry and stage router.
- Keep behavior compatible through the old environment/CLI aliases.

### 3. Migrate persistence and progress tracking

- Add schema versioning.
- Add source identity/revision and distillation provenance.
- Add the consolidation ledger.
- Replace destructive `INSERT OR REPLACE` paths.
- Migrate a copy of the real DB and verify row counts, FTS, statuses, and idempotence.

### 4. Add transcript-source adapters

- Move existing Claude parsing behind `claude-jsonl` without behavior change.
- Implement `codex-jsonl` from fixtures.
- Verify that source selection is independent from model routing.

### 5. Add provider adapters

- Refactor the existing Codex wrapper to the neutral contract.
- Make usage nullable and validate output locally.
- Adapt Claude lazily as an optional compatibility provider.
- Configure Codex CLI for both stages on this host.

### 6. Fix consolidation batching

- Select explicit unconsolidated distillation identities.
- Split by actual serialized budget.
- Persist ledger entries only for successfully processed identities.
- Prove that a multi-batch backlog is completely drained without loss or duplication.

### 7. Add conflict-safe review

- Persist base hashes.
- Add JSON listing and granular resolution commands.
- Preserve configurable nightly auto-apply for this installation.
- Test concurrent/manual memory edits between suggestion and apply.

### 8. Add the Codex skill and provider-neutral automation

- Create and validate the skill metadata and instructions.
- Update install logic and systemd examples.
- Remove provider/model claims from core README sections and prompts.

### 9. Cut over on copies, then live data

- Run the full pipeline against copied DB and memory directories.
- Import one historical and one current Codex session.
- Run distill and consolidate through Codex CLI subscription auth.
- Exercise conflict, timeout, invalid JSON, and unavailable optional-provider paths.
- Cut over the live timer only after all acceptance criteria pass.

## Acceptance criteria

- The full configured pipeline works with no `claude` executable in `PATH`.
- Core tests run without Codex, Claude, network access, or credentials through a fake provider.
- Changing transcript source does not change model provider; changing model provider does not change transcript source.
- A normal Codex root session is imported; subagent and guardian sessions are rejected.
- AGENTS, environment context, developer messages, commentary, reasoning, tools, and events do not enter the distilled transcript.
- Resume and compaction do not duplicate visible conversation history.
- A historical Codex session older than the former cursor is consolidated exactly once.
- A backlog larger than one batch loses no distillation and produces no duplicate ledger consumption.
- Provider timeout, invalid JSON, schema failure, or DB failure leaves all affected distillations unconsolidated.
- Every stored distillation identifies source revision, parser version, prompt version, provider, and model.
- Missing Claude CLI does not affect status, ingest, search, or Codex-backed stages.
- Codex CLI usage is `null` when unavailable, never fabricated as zero.
- A memory file changed after suggestion creation causes a conflict instead of overwrite/delete.
- Migration preserves current sessions, messages, FTS search, distillations, suggestions, statuses, archived memories, and memory files.
- Re-running migrations and ingest is idempotent.
- The Codex skill contains no hard-coded provider or model routing.
- New installs default to suggest-only; this host preserves its explicitly configured auto-apply behavior.

## Advisor review disposition

An independent advisor reviewed the architecture before this plan was written.

Accepted findings:

- Replace timestamp cursoring and silent batch truncation with an explicit consolidation ledger.
- Separate stage routing from the provider execution contract.
- Add source identity/revision, parser version, and distillation provenance.
- Add apply preconditions and granular review commands.
- Make configuration strict and adapter loading lazy.
- Preserve nullable provider usage and validate structured output locally.

Amended findings:

- Codex session filtering uses both observed fields: `payload.source` and `payload.thread_source`. The advisor's claim that `thread_source` is absent did not match the current local Codex 0.147.0 files.
- Nightly auto-apply is not removed during this migration because it is an intentional current behavior. It becomes an explicit review mode with hash preconditions; new installations remain suggest-only by default.

## Implementation record

- The live database was migrated additively after creating `~/.claude/dream.db.bak-provider-migration-20260807T222042Z`.
- Codex ingest added 90 sessions and 946 filtered messages; legacy Claude history remains available through its optional source adapter.
- Both model stages route through the Codex CLI by configuration. A synthetic structured-output call passed; no private transcript was sent during validation.
- The installed Codex skill, user configuration, systemd unit, schema migration, provider/source adapters, review conflict checks, and consolidation ledger were validated.
- Final automated validation: 64 tests passed, Python compilation passed, Pyright reported zero errors, skill validation passed, and `git diff --check` passed.
