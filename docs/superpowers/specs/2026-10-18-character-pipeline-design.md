# Character pipeline design

## Status

Approved for implementation on 2026-10-18, with the decisions below.

Date: 2026-10-18. Base: `ec3ff2f` on fresh `main`, including merged PR #1.
Branch: `feat/character`. Szymon approved implementation and the plan without another review stop. Real
provider calls on real transcripts, local config edits, push, and publication
still require a separate request.

## Goal and scope

Dream will build character preference data and deploy approved examples through
context. The model proposes, the host enforces, and the human decides. This
iteration changes no weights. The constitution is human-owned. A model verdict
never accepts a pair. Existing memory suggestion curation cannot touch this queue.

The intended result is a human-approved constitution, correction and synthetic
pairs with full provenance, a manual review queue, a deterministic context file,
a baseline versus character evaluation, and an export for later training.
Retention exclusions govern every stage, including artifacts made before the
policy changed. Python 3.11+ standard library remains the entire runtime.
pytest remains a development dependency. No new service or HTTP client is needed.

## Research and limits

[Claude's constitution](https://www.anthropic.com/news/claude-new-constitution)
explains values with reasons and uses them to create synthetic data and rankings.
[Claude's character](https://www.anthropic.com/research/claude-character) describes
character training as part of post-training. These support using explicit traits,
reasons, and conflict priorities, rather than deriving traits from user corrections.

[Open Character Training](https://arxiv.org/abs/2511.01689) and its
[implementation](https://github.com/maiush/OpenCharacterTraining) use constitutional
data and weight training, including DPO and introspective SFT. Dream borrows the
data flow. Context deployment is a separate experiment. It does not establish the
same robustness as weight training, and the evaluation must not claim that it does.

[Persona vectors](https://www.anthropic.com/research/persona-vectors) studies
internal model representations for character monitoring and control. That needs
model internals which Dream's CLI providers do not expose. It stays future work.

## Current code and integration points

- `config.py` merges packaged defaults, local TOML, existing environment settings,
  and CLI overrides. Provider adapters load lazily through `providers/registry.py`.
- `backend.generate` builds `GenerationRequest` and returns `GenerationResult`.
  Its current system prompt is specific to memory distillation and JSON output.
- `sessions` and `messages` already contain visible turns. Turn identity is
  `session_id` plus `seq`. `sessions.source_revision` and `parser_version` identify
  the ingested revision. No character command needs to open source transcript files.
- `dream.py` owns CLI dispatch, writable database opening, additive migrations,
  backups, and guarded memory writes. `schema_migrations` currently reaches 2.
- `storage.open_db_readonly` refuses an active nonempty WAL. New read commands use
  this helper and report that exact error, without opening a writable connection.
- `retention_policy.py` supplies `load_retention_policy`, `context`, project and
  text checks. `retention_cleanup.py` snapshots affected records before cleanup.
- `run_context` accepts `memory_root`; `render_context` accepts `query_text`.
  Existing recall remains relevance-ranked, untrusted reference data.
- `metrics.STAGES` is a fixed tuple of four stages. `finish_run` and `seed_success`
  name `curation` as the last stage. `dream/systemd/dream.service` wraps every stage
  with `dream-stage --continue-on-error`. It contains no literal provider command. Provider selection belongs in config.

## Architecture and alternatives

I recommend one focused `dream/character/` package, separate SQLite tables, and
one `dream character` command group. Reuse provider requests, configuration,
retention checks, and database backup machinery. Keep character states separate
from memory `suggestions` and never index character data in `recall_documents`.

An alternative is to reuse `suggestions`. That would share file-oriented statuses
and auto-apply paths with a queue that must require human acceptance. It would
also mix pair provenance with memory target paths. I do not recommend it.

A file-only queue is another option. It would avoid new tables, but would need a
second mechanism for concurrent review, retry identity, retention snapshots, and
evaluation history. SQLite already handles these jobs in Dream.

The proposed flow is:

```text
human constitution + ingested visible turns / human seeds
    -> deterministic eligibility and retention filters
    -> extraction / generation -> pending pair revisions
    -> two independent-order judge calls -> advisory verdicts
    -> explicit human edit / accept / reject / defer
    -> current-policy eligibility -> render / DPO export

fixed held-out probes + baseline / rendered context
    -> response generation -> rubric judge -> stored comparison
```

### Proposed modules

| Module | Responsibility |
| --- | --- |
| `character/constitution.py` | Parse and validate human document, stable trait and seed IDs, approval and content identity. |
| `character/types.py` | Typed records, strict provider schemas, semantic validation, prompt versions. |
| `character/storage.py` | Queries, unique work keys, revision checks, item transactions, review events. |
| `character/safety.py` | Protected identities, allowed output paths, current-policy eligibility, backup and atomic replacement checks. |
| `character/generate.py` | Bounded correction windows, extraction, seed expansion, paired responses, estimate. |
| `character/judge.py` | Family checks, swapped-order calls, normalized verdicts. |
| `character/review.py` | Human-only state transitions and explicit chosen-text edits. |
| `character/render.py` | Pure deterministic selection and formatting of approved inputs. |
| `character/eval.py` | Fixed probe runner, rubric scores, drift records, comparison reports. |
| `character/targets.py` | Target protocol and the baseline and A implementations. |
| `character/export.py` | Stable DPO JSONL serialization of eligible accepted pairs. |
| `character/cli.py` | Command parsing helpers and orchestration, registered by `dream.py`. |

`config.py`, `default-config.toml`, `schema.sql`, `dream.py`, `backend.py`,
`install.sh`, `retention_cleanup.py`, and `README.md` receive narrow integration
changes. `recall_documents.py` and consolidation input readers receive protected
source exclusion checks. Optional hook and nightly changes depend on Open decisions.
No compatibility aliases, legacy formats, or dual storage paths are proposed.

## Shared contracts

### Configuration

Add a strictly validated `[character]` section. It never reads `review.mode`.
The proposed keys are listed below. There is no review-mode key because human
review is the only mode. Unknown auto-apply and review-mode keys fail validation.

| Key | Contract and proposed value |
| --- | --- |
| `enabled` | Boolean, false in packaged defaults. Enables character writing/model commands, not automatic execution. |
| `constitution_path` | Nonempty path, proposed `~/.config/dream/character/constitution.md`. Approved outside the memory root. |
| `state_root` | Nonempty path, `~/.local/state/dream/character/`, outside memory and protected instruction directories. |
| `allow_same_family_judge` | Boolean, false. An explicit override must appear in provenance and reports. |
| `hook_injection` | Boolean, false. Implemented for session-start only. |
| `render_max_pairs` | Positive integer, rejects booleans. Approved default 3. |
| `render_budget_codepoints` | Positive integer covering the whole rendered block. Approved default 12000. |
| `max_context_codepoints` | Positive integer, proposed 12000 per original task window. |
| `max_response_codepoints` | Positive integer, proposed 8000 per candidate response. Oversized complete records defer. |
| `max_seed_expansions` | Nonnegative integer, proposed 0. Expansion is explicit, bounded, and separate from trait creation. |

Each new model stage uses existing stage fields, with strict types and positive
timeouts: `character_extract`, `character_generate`, `character_judge`, and
`character_eval_response`. Proposed packaged routes select `codex`, `codex`,
`claude`, and `codex`, respectively. New stage defaults contain provider routing
only. Model names, reasoning options, timeouts if customized, and enabled sources
belong to local configuration. Existing generic timeout behavior remains usable.
No local configuration is edited during implementation.

Character commands require explicit model IDs in local stage settings before
model calls, so provenance and family checks do not depend on an unknown CLI
default. Store configured and reported model identities. If a provider cannot
report its resolved model, label it as configured but unverified. Reject a reported
family mismatch. No model IDs are embedded in core code.

Character types, keys, and route references fail during configuration validation.
Family checks apply when character is enabled. Executable/auth preflight remains
scoped to selected commands: disabled character judging must not make existing
Codex-only commands require Claude. Update `backend.preflight` to exclude inactive
character routes and support explicitly requested character stage checks.

### Provider requests and output validation

Extend `backend.generate` with an optional `system_prompt` parameter, forwarded
to the existing `GenerationRequest`. Existing callers keep their current prompt.
Character passes a provider-neutral task prompt and schema. Response generation
returns `{"response": "..."}`. Every model payload uses `additionalProperties:
false`, required fields, enums, and bounded host validation.

Do not rely on unsupported JSON Schema keywords. `model_types.validate_output`
currently checks types, required keys, enums, and extra properties. Character
validators separately enforce length, nonempty text, valid trait IDs, source
identities, score ranges, and exact candidate labels.

Every character prompt appends the current `policy.context()`, including seed
expansion, synthetic baseline, response generation, judgment, eval response, and
eval scoring. The synthetic baseline omits constitution text, trait explanations,
chosen output, and constitutional instructions. The policy remains in both arms.
English instructions require English generated labels, rationales, and reports.
Original quotations and constitution bytes remain verbatim. Regenerated responses
keep the original moment's language. Synthetic answers follow the seed's language.

### Additive schema and provenance

Add migration version 3. The migration backup check must detect a database below
version 3 before `executescript` creates any character tables. Use SQLite backup
for an existing database. Record version 3 only after successful creation. Never
replace existing tables or FTS indexes. Fresh databases need no pre-migration copy.
Read commands report missing character schema and never migrate it.

All derived rows carry `constitution_sha256`, `constitution_version`,
`prompt_version`, and UTC `created_at`. Prompt version is a content hash of the
specific template and schema version. A rendered file, export, and report carry
the same identities in their headers or metadata. Proposed tables are:

| Table | Main columns and constraints |
| --- | --- |
| `character_constitutions` | SHA primary key, human version, approval timestamp, observed timestamp. Metadata only, no writable constitution copy. |
| `character_moments` | ID, unique work key, session ID, assistant and correction seq, source revision/parser version, project slug snapshot, extraction disposition, is_correction/context_sufficient flags, original context messages/hash and window status, assistant text, correction text, trait IDs JSON, agreement enum, rationale, extraction provider/model/options/usage/duration, content hash, revision, excluded flag, shared provenance. |
| `character_seeds` | ID, human seed ID/hash, trait ID, optional parent expansion ID, prompt messages, origin human or expanded, generator provenance for expansion, content hash, revision, excluded flag, shared provenance. No public evaluation seed allowed. |
| `character_pairs` | ID, work key and generation_attempt with a joint unique constraint, supersedes pair ID, source kind, moment or seed ID, project/source identity snapshot, prompt messages, chosen, rejected, rationale, trait IDs, conflict flag, generator call provenance JSON for both arms, content hash, revision, review status, edited flag, excluded flag, shared provenance. |
| `character_judgments` | ID, pair ID/revision/hash, two ordered verdict objects, normalized result, judge provider/model/family/options, override flag, usage/duration, shared provenance. Unique completed result per pair revision and judge/prompt identity. |
| `character_review_events` | ID, pair ID/revision/hash, human action, reason, old/new review state, timestamp, shared provenance. Host marks origin as human CLI, never from provider output. |
| `character_eval_runs` | ID, probe/rubric hashes, baseline and A target identities, render hash and accepted pair revision manifest, response/judge routes, override flag, run state, dependency manifest, shared provenance. |
| `character_eval_results` | Run ID, probe ID/category, target, multi-turn responses, response/scoring state, bounded failure code, both swapped scoring payloads and normalized score objects, inconsistency flag, rationale, call provenance, content hash/revision, excluded flag, shared provenance. Unique run/probe/target result. |

Generator call provenance includes route name, provider type/auth class, configured
and reported model, family, stage options, start/finish times, duration, usage
when available, and prompt hash. Never store credentials. Original sources retain
session ID and turn seq, project snapshot, source revision, and parser version,
even if a later ingest replaces the current `messages` rows. Synthetic sources
retain seed identity and expansion lineage. No generated table enters recall FTS.

Work keys hash source identity/revision, constitution SHA, stage prompt versions,
provider/model/options, and context/response bounds. A completed item is
idempotent. Failed calls do not
insert completed rows or advance any cursor. There is no global timestamp cursor.
A successfully extracted moment may remain when its later response call fails.
It is valid extraction work, not a completed pair. A later run retries that pair.
Synthetic pair arms commit together. Swapped judgments commit together.

### Approval, staleness, and concurrency

If the constitution itself matches a current retention marker, refuse model work,
render, export, and eval with a concise policy-conflict error. Do not print the
matching body, rewrite it, or weaken the policy. Human editing resolves this
conflict. List/show can report the conflict without leaking the excluded text.
Check the
complete rendered block too, including its example explanations and headers,
before eval sends it to a provider or any deployment command emits it.

Review status is `pending`, `deferred`, `edited`, `accepted`, `rejected`, or
`stale`. `edited` means human text exists but still needs explicit acceptance.
Conflict, judge inconsistency, and retention exclusion are separate fields.

The current constitution SHA determines eligibility on every read. Old unreviewed
pairs appear as stale immediately, without a database write from list/show.
The next successful character mutation persists that state and invalidates old
judgments. Accepted old-SHA pairs remain in history but cannot render or export.
Returning to old constitution bytes does not silently revive a pair already
marked stale. Reconsideration requires `generate --refresh` and fresh acceptance.
Refresh creates a new pair row with a host-incremented generation attempt and
supersedes ID. It preserves old pair and review history. Normal generation skips
completed work keys, including rejected pairs. Explicit refresh also allows a new
attempt after rejection and never overwrites an accepted or rejected row. The
estimator reports which old pair each fresh attempt supersedes.

Every mutation snapshots pair revision/hash, constitution bytes, source revision,
and policy fingerprint. Immediately before commit, reload the policy and
constitution, then compare all preconditions. Changed inputs defer that item and
report it for retry. Review commands take `--expected-sha` copied from `show`.
Accept binds to the exact displayed revision, full candidate text, and constitution.
A changed chosen response invalidates its judgments and acceptance.

### Write authority and backups

Model output never supplies a path, SQL statement, row ID, approval status, or
trait definition. Host IDs bind outputs to inputs. Rendered blocks, exports, and reports go under
the validated state root. Database and retention backups use their existing
protected locations. Render, export, and report filenames are host-generated;
there is no arbitrary `--output` path. `edit --body-file PATH` reads a file and
updates SQLite only. It never writes that file or invokes an editor.

Before opening any writable database or output, check configured database, state,
backup, and temporary-file paths against the constitution, retention policy,
`AGENTS.md`, `CLAUDE.md`, and native memory directories. Resolve `..`, symlinks,
parent directories, and existing inode aliases. Refuse protected destinations,
including the database itself pointing to a protected file. Recheck before
replacement or deletion and use exclusive temporary files. Never follow a
symlink at an output leaf. The installer also refuses symlink targets.

Every successful command that changes existing character rows makes a consistent
SQLite backup before its first mutation. File replacement backs up existing
bytes first. Backups live outside memory, have owner-only permissions, and never
enter recall or model prompts. Provider work happens before opening a writer,
using a read-only snapshot of existing inputs. New database setup is explicit
through a writing command. A failed provider item creates no backup or mutation
for that item. Healthy completed items may still commit, as required by isolation.

## M1. Human-owned constitution

`character/constitution.py` parses a deliberately small Markdown format with
frontmatter fields `status`, `version`, and `approved_at`. Use a strict line parser
for string scalars, not YAML or a new dependency. Unknown/duplicate fields fail.
Status is exactly `draft` or `approved`; version is nonempty; approved status
requires an ISO 8601 approval timestamp. The host never changes these fields.
SHA-256 covers exact UTF-8 file bytes, including frontmatter.

Body sections are `Traits`, `Priority order`, `Hard constraints`, and `Seed
prompts per trait`. Traits use stable human IDs such as `trait-01`, first-person
text, and a `Why:` line. Seeds have unique human IDs and an explicit trait ID.
About ten traits is guidance, not a hard count. Approval validation rejects empty
required sections, unresolved required-section template slots, missing reasons, unknown trait
references . Seed prompts are optional for approval. Empty seed TODO slots do not block
approval. Only Traits with Why lines, Priority order, and Hard constraints are
required. Synthetic generation and expansion skip seedless traits and report a
clean skip, including `--source all`. All other character commands work without
seeds. Structural validation
cannot prove that a statement is first-person or that the human wrote it.

After approval of this design, ship
`dream/character/constitution.template.md`. It starts in draft and includes exactly
these draft example traits, verbatim:

- `trait-01`: "Jeśli coś da się sprawdzić deterministycznie, nie zostawiam tego własnemu osądowi, bo kod nie dryfuje."
  Reason already present in the seed: "bo kod nie dryfuje."
- `trait-02`: "Wolę dopisać niż przepisać czyjąś przemyślaną pracę. Przepisuję dopiero wtedy, gdy jest wyraźnie nieaktualna."
  `Why: TODO`
- `trait-03`: "Kiedy coś zawodzi, przekazuję dokładny błąd, zamiast zgadywać naprawę, bo zgadywanie ukrywa przyczynę."
  Reason already present in the seed: "bo zgadywanie ukrywa przyczynę."

The template uses `Why:` lines for all three. It has clearly labeled empty TODO
slots for traits 04 through 10, priority order, hard constraints, and human seed
prompts. These TODOs are required human work, not model-generated values. Dream
never invents further traits or priority rules. `install.sh` copies the template
to the live path only if it does not exist, without overwriting any existing file.
This one-time installation creates a draft. Runtime commands never write it.

Generate, judge, render, export, and eval require an approved constitution before
provider work or writes. List/show can explain a draft and stale history safely.
The live document stays outside memory by recommendation, as approved.
Reject resolved references to it in consolidation input and recall synchronization
and reads, including symlinks placed in memory and already indexed aliases.
Do not add it as an approved-memory document.

Tests in `test_character_constitution.py`, `test_character_safety.py`, and
`test_character_migrations.py` cover parsing, draft refusal, human-only approval,
copy-once install, exact Polish text, SHA changes, stale reads without writes,
backup-before-migration, and protected path/inode aliases for every writer.

## M2. Data factory

`dream character estimate --source corrections|synthetic|all --limit N` performs
no model calls or writes. `dream character generate` accepts the same source and
limit flags, plus explicit `--refresh`. Estimate accepts `--refresh` too and
reports the new attempts it would create. Require a positive explicit limit for generation. Stable order is
session ID plus correction seq, then trait ID plus seed ID. Limit counts source
work items, not successful results, so failures cannot cause unbounded extra calls.

The shared estimator and generator select exactly the same eligible work list.
Estimate reports source items, bounded character counts, routes/models, and call
bounds. A correction needs extraction and at most one regeneration call. A seed
needs two responses, plus any explicitly requested expansion calls. Judging costs
two calls per pair and is shown separately. Never claim an exact token cost.
Seed expansion is a separate bounded phase in `generate --expand-seeds`; estimate
shows its bound without pretending to know future prompt text.

For corrections, query visible user and assistant rows only. A deterministic,
Unicode-aware, case-folded token/phrase matcher checks English and Polish cues
such as `nie`, `nie tak`, `źle`, `cofnij`, `stop`, `don't`, `actually`, and `that's
wrong`. Require an immediately preceding assistant response. Cues only select
candidates. They are not proof of a correction.

Build a complete-message suffix ending before that assistant response, within
`max_context_codepoints`, plus the complete response and following user turn.
Require the initiating user message. Exclude deterministic oversized or missing-
context items from the eligible work list before applying `--limit`, and report
their counts and safe identifiers. They cannot consume every later run's limit.
An LLM-detected insufficient-context result is a completed extraction disposition
under the source/config work key and is skipped until inputs change or refresh
is requested. Provider errors remain retryable, with no completed disposition.
Remove excluded projects before building prompts. If a
selected window, response, or correction matches text markers, drop the entire
moment before any call. Do not stitch text across a removed turn.

Extraction uses `prompts/character-extract-v1.md`. Strict JSON has `is_correction`,
`context_excerpt`, `corrected_assistant_response`, `user_correction`, `trait_ids`,
`agreement` as `agrees|conflict|unclear`, `context_sufficient`, and `rationale`.
Host checks copied text
against the supplied messages and validates trait IDs. Provenance and IDs come
from the host. A validated non-correction stores only a completed extraction
disposition and its work key, so later runs do not call the extractor again for unchanged inputs. It
creates no pair. A conflicting correction remains visible with `conflict=true`; no model can resolve that flag
into human acceptance. Filter extraction output again before persistence.

Proposed regeneration rule: the model does not see the later user correction,
its extracted rationale, or agreement label. It sees the bounded visible task
messages preceding the rejected response, plus the approved constitution. Thus
the pair assigns both answers to that reconstructed task prompt. It does not claim
identical original inputs. The regenerated model cannot copy
new facts from a correction unavailable in that prompt. This is a reconstruction
from visible turns, not the full original tool, system, or repository context.
Extraction must mark insufficient-context moments ineligible and report their
count. Store the exact reconstruction hash, message range, omitted-message count,
and context status in pair provenance. Judgment and export reuse those same
messages. The rejected text is the original assistant response. The chosen label means proposed preference until
human acceptance. Equal or empty responses fail pair validation. Correction
conflicts remain review evidence, not a new trait or instruction.

Synthetic seeds come only from approved human seed sections. Optional expansion
uses `prompts/character-expand-v1.md`, keeps the original trait ID, and stores
parent seed provenance. It cannot output traits. Generate chosen and rejected
through the same configured provider/model/options with fresh independent calls.
Only the chosen arm receives the constitution. Never leak chosen output into the
baseline call. Use `prompts/character-response-v1.md` with explicit arm framing.

Filter both arms, metadata, rationales, and source project snapshots before
writing. Every prompt treats transcript excerpts, seeds, and candidate text as
untrusted data. No instructions in those fields can approve a constitution or
pair. Log each bounded provider/schema error by work ID, continue remaining work,
and exit 1 if any item failed. A Dream database error is fatal and rolls back the
current item. Interruptions propagate. Never catch storage errors as model errors.

README privacy changes list exact flows: extraction sends bounded original
context, assistant response, correction, and constitution to the extraction route;
regeneration sends original task context and constitution to the generation route;
synthetic baseline sends seed task and retention context; synthetic chosen also
sends constitution. Stored correction text goes to the judge as separately labeled
evidence, never as authority. Human review, render, and export are local.

Tests in `test_character_generate.py` cover multilingual cues and false positives,
visible-only turns, equal prompt context, no correction leakage in regeneration, successful negative extraction caching,
insufficient-context skipping, conflict preservation, strict extraction copying, seed lineage, route identity,
provenance, estimate agreement, limits, refresh after staleness, retries, and mixed
item failures. Reuse
fake `GenerationResult` and monkeypatch patterns from provider and ingest tests.

## M3. Independent judge

`dream character judge --limit N` loads eligible unjudged current pair revisions.
`character/judge.py` maps configured provider type/auth to a family. A
`codex-cli` route with `chatgpt-subscription` is `openai`; `claude-cli` is
`anthropic`. Unknown auth/family combinations fail. Different route names or model
names within one family do not satisfy independence. Validate against the configured regeneration
or synthetic response model family, and against the eval response family when
evaluating. The original rejected session response may have unknown provider/model
identity because the ingest schema does not store it. Preserve any source-derived
identity as evidence and otherwise record unknown. Do not claim judge independence from that historical response. For
`sessions.source = "claude"`, treat the source as reliable evidence of an
Anthropic-written original response. When the judge is Anthropic, show the overlap
warning in review, export, and eval provenance. Other unknown sources stay unknown.
If reliable source evidence shows overlap between the historical model and judge,
display that overlap as a warning in review and export provenance. It does not
replace the required independence from Dream's regeneration model.

Judging runs nightly through configuration after generation. Same-family use
requires the explicit false-by-default override. Record it in both verdicts,
review display, exports, and eval reports. The systemd unit contains no literal provider commands. Config validation does not execute a provider.

Use `prompts/character-judge-v1.md`. Provide the approved constitution, task
context, anonymously labeled response A/B, and separately marked correction
evidence. Hide proposed chosen/rejected labels and generator identities.
JSON fields are `winner` as `A|B|tie|neither`, `assessments` with one object per
implicated trait containing `trait_id`, `a_assessment`, `b_assessment`, and
`rationale`, plus a nonempty overall `rationale`. Assessments use fixed enums
`supports|violates|unclear`. Validate exact trait coverage and all labels.

Run twice with A/B swapped. Normalize winners and per-response assessments back
to pair identities. Any disagreement in winner or assessment labels marks the
result `inconsistent`; natural-language differences alone do not. Agreement on
`tie` or `neither` remains that verdict. A failed second call saves no completed
judgment and leaves the item retryable. Models never swap the stored pair text
or make a human decision. Successful judgments bind to the exact pair revision.

`test_character_judge.py` covers family routing/auth errors, explicit override,
unknown historical models, mixed-family correction pairs, swap normalization, inconsistent assessments, ties, schema errors, failed second
calls, revision races, English rationale instructions, and no auto-accept.

## M4. Human review queue

Commands are `dream character list`, `show ID`, `accept ID --expected-sha HASH`,
`reject ID --expected-sha HASH`, `defer ID --expected-sha HASH`, and
`edit ID --body-file PATH --expected-sha HASH`. `--reason TEXT` is required for
accept/reject/edit. No provider runs during review. No batch accept or auto-apply
command exists. No nightly path accepts or edits pairs.

List/show open the database read-only and filter with the current policy before
printing content, provenance, or rationales. Show includes both texts, source
identities, constitution identity, pair hash/revision, every judge verdict,
rationales, and conflict/inconsistency/edit flags. Hidden pairs cannot be accepted.
Rejected, stale, and excluded records cannot transition directly to accepted.

Acceptance requires a complete judgment for the displayed pair revision. Conflict,
inconsistency, or a verdict against chosen requires explicit `--acknowledge-flags`
and a human reason; it never silently blocks the human from deciding. The stored
review event captures which flags the human acknowledged. Defer preserves the
pair for later review. Reject ends it. Edit creates a new revision, marks `edited`,
clears acceptance and judgments, and requires rejudging then explicit acceptance.
Accepting an edited pair keeps its edited provenance. Editing never changes traits,
constitution bytes, rejected text, or original source evidence.

`test_character_review.py` covers all transitions, stale and concurrent hashes,
flag acknowledgement, edit invalidation, backups, policy changes, no provider calls,
and auto-apply config rejection. Extend `test_provider_architecture.py` to assert
that the unit never invokes character accept/edit or an equivalent auto-review path.

## M5. Deployment A through a rendered file

`dream character render --preview` prints a read-only preview. Without preview,
write a host-named `block.md` and a JSON manifest under the state root, with backups
and atomic replacement. It never installs content into any instruction file.
Szymon chooses where to load it. Merely creating the file changes no session.

`character/render.py` consumes the approved constitution plus eligible accepted
current-SHA pairs. Select at most K pairs within the total Unicode codepoint
budget. Keep the entire constitution verbatim. If it plus the fixed header does
not fit, fail rather than truncate it. Greedily choose the pair covering the most
uncovered trait IDs, break ties by ascending pair ID, then repeat. Format each
whole example before selection; skip examples which do not fit. No partial pair
or omitted `Avoid` text is allowed. Headers count toward the budget.

The block header includes constitution SHA/version and selected pair IDs/revisions.
Each example has bounded task context and `Preferred`, `Avoid`, and `Why` labels.
Why uses the human acceptance reason. Examples are quoted data; the host never promotes their embedded
instructions to constitutional rules. Quoting and typed section separation can
be tested deterministically. Resistance to a model following quoted attacks is
an evaluation goal, not a guarantee from formatting. Keep preferred/avoid text exact.
Identical eligible inputs and settings produce identical UTF-8 bytes. No current
time, absolute home path, or run ID appears in the rendered block.

The manifest records policy fingerprint, pair hashes, and block hash. Recheck policy
and accepted revisions before output replacement. A manually copied block cannot
be revoked by Dream after retention changes. Document that limitation and require
rerendering after a policy or constitution change. Dream-managed files are rebuilt
or invalidated by retention cleanup. Hook mode must never trust a cached block
without validating its full current-policy manifest.

Approved hook design: compose a separate trusted
character section in `cmd_context` on session-start events, including compact.
Build it from human-approved sources through the same pure renderer. Keep
`run_context(..., memory_root=...)`, query relevance, and
`render_context(..., query_text)` unchanged for untrusted recall. Recall candidates
cannot be arguments to the trusted renderer. A common result composer joins typed
sections after independent rendering and budgets. Respect the hook's total context
limit and five-second timeout; do not truncate the constitution. If the complete
character section cannot fit, omit it with a safe diagnostic. No provider or DB
write runs in a hook. Failure omits character and keeps normal recall available
when safe. Character injection remains off by default and requires explicit local
configuration and review of any changed hook definition.

`test_character_render.py` covers deterministic bytes, coverage ties, complete pair
budgets, approval/edit/SHA/policy filtering, no timestamps, and output backups.
`test_character_security.py` covers quoted instruction attacks, protected aliases,
and the inability to promote recalled approved-memory text, distilled summaries,
or raw transcripts into the trusted section. These tests prove host data flow,
not how any model will respond to the resulting text. Extend context and hook
tests for compact, budget limits, fail-open behavior, and separation.

## M6. Fixed-probe evaluation

`dream character eval --fixtures dream/fixtures/character/public.json --limit N`
performs explicit model work and persists a run. `--report RUN_ID` only reads and
prints the current-policy-filtered JSON and Markdown table. Evaluation requires
an approved constitution and an explicitly configured response and judge model.
No live transcript smoke test occurs during development.

Fixtures contain stable probe IDs, category, applicable trait IDs, visible
multi-turn messages, a fixed rubric with score anchors, and expected behavioral
checks. Categories cover traits, attempts to break character, user wishes that
conflict with the constitution, and unsupported user pressure across turns.
Include neutral task controls so the report can show a loss in task usefulness.
Constitution-specific fixtures bind a trait ID to a hash of its exact trait text
and Why line, plus the applicable rubric version. If a trait is missing or its
meaning has changed, report `unsupported` and skip that probe without model work.
A changed ID is not silently rebound. Generic pressure and task-control probes
use constitution-relative rubrics without asserting a particular personal trait.
Report unsupported counts separately from failures and never invent a score or
rewrite public fixtures to improve applicability.

Public fixtures never enter extraction, seed expansion, pair generation, example
selection, or threshold selection. Assert fixture identity disjointness and exact
content hashes. Private calibration probes remain separate and are not optimized
using public results.

`character/targets.py` defines a target protocol with `identity()` and
`respond(messages, context_block, request_settings)`. Baseline and A share the
same response route/model/options, structured response schema, and probe messages.
Only A prepends the exact rendered character block. Keep the A manifest fixed for
the run. Future B can implement this protocol through a separate adapter. No B
endpoint, HTTP transport, weights, or local model process ships now.

For multi-turn probes, each target sees its own previous responses followed by
the same scripted user turn. Judge complete trajectories against the fixed rubric
and constitution. Hide target labels and generator identity. Score both trajectories
with swapped order, store both judgments, and report inconsistent scores separately
rather than averaging them into a success. Use `prompts/character-eval-v1.md` for
response instructions and `prompts/character-eval-judge-v1.md` for scoring. Scores
are host-validated integers on an anchored 0 through 4 scale per rubric dimension.
Also record deterministic protocol failures such as empty answers.

Reports show per-category baseline score, A score, delta, valid sample count,
inconsistent, unsupported, and failed counts, and task-control scores. Baseline/A
deltas use only the identical probe subset successfully generated and consistently
scored for both targets. Show the denominator and omissions for every category.
Store prompt, fixture,
rubric, render, constitution, route/model identities, and usage for comparisons
over time. Comparisons across different fixture or rubric hashes are labeled
noncomparable. Model-judged scores are evidence, not proof of reliable character.
A report with failures is incomplete and the command exits 1. Complete independent
probe/target responses may persist; a failed result never counts as scored. Store
a response as `generated` before scoring, then `scored`, `inconsistent`, or
`score_failed`. A failed generation has a separate bounded failure record, no
validated response, and no success cursor. A failed second scoring call marks the
score attempt failed and never stores a completed score from the first call alone.
Retry both scoring orders. Completed scores retain both validated payloads.
Failure records contain safe codes and call identity, not raw provider output.

Filter fixtures before calls and responses, rationales, and metadata before writes
and again for report reads. If a rendered pair becomes excluded, hide dependent A
results and their aggregates, even when their generated text lacks the marker.
Never publish automatically. Generated report files stay in the protected state
root; publication requires a separate user request and privacy review.

`test_character_eval.py` uses fake targets/judges only. Cover paired route identity,
multi-turn pressure, hidden labels, fixture separation, rubric ranges, incomplete
runs, failed second scoring calls, partial target failure, paired score subsets,
unsupported trait bindings, swapped-score inconsistency, drift identities,
read-only reports, and
retention propagation through the rendered manifest.

## M7. Export for future training

`dream character export --format dpo-jsonl --preview` prints a read-only preview.
Without preview, it writes host-named `accepted.dpo.jsonl` under the state root.
Read the source through `open_db_readonly`; this command writes a file, not SQLite.
Only accepted, current-constitution, current-policy, exact accepted-revision pairs
qualify. Read eligibility is the same function used by render.

Each line has `prompt` as the original task message array, `chosen` and `rejected`
as assistant strings, and `metadata` with constitution SHA/version, pair ID/revision,
human review reason/edited flag, judge/override flags, prompt hashes, and source
provenance. Keep the later correction in provenance evidence, outside the task
prompt. Metadata labels correction prompts as bounded visible-context reconstructions
and includes their context hash and omissions. The constitution is recorded by
identity, not silently inserted into the DPO task prompt. This defines a versioned `dream-dpo-v1` format; a future trainer
can map it explicitly to its own input format. It is not a promise that every
trainer accepts identical JSON fields.

Serialize UTF-8 JSON with stable key ordering and pair ID order. No training code,
model download, or remote call occurs. Check policy and protected output identities
before writing and again before replacing. Existing exports get backups first.
`test_character_export.py` checks exact field structure, provenance, edits, stable
bytes, policy changes, excluded-source lineage, symlinks, and accepted-only output.

## M8. Retention cleanup

Extend `dream retention` preview with affected character moments, seeds, pending
pairs including deferred/edited/stale rows, accepted pairs, judgments, review
events, eval records, and managed output files. Project identity snapshots and
source lineage remain available even if sessions change or disappear.

`character/safety.py` centralizes eligibility. Apply current project/path/text
checks to every textual field, including rationale and nested metadata. Excluding
a source moment/seed excludes its pairs, judgments, review payloads, exports,
rendered manifests, and dependent evaluation results. Prose-only semantic
exclusions remain prompt guidance; deterministic matching has the same limits as
existing retention. Never claim paraphrase detection without model judgment.

Use the existing retention backup directory and `affected-records.json`. Preview
counts affected moments, expanded seeds, pairs in every review state, judgments,
review events, eval results, and managed files. Human seeds live in the constitution
and are never changed. Back up every affected row and managed file byte, including
human review reasons. Then delete the affected derived rows and all their lineage
dependents. Do not retain scrubbed payloads or audit tombstones in live tables.
Original sessions and messages remain unchanged.

Inside the character cleanup write transaction, recompute the complete affected
set and its content hashes. If it differs from the backup snapshot, roll back the
whole character part, report the conflict, and ask for a rerun. There is no per-row
partial character deferral. Existing memory cleanup semantics remain unchanged.
Delete or invalidate matching managed files, so rerender or re-export is required.
Leave user-modified files in place and report them. Recheck protected identities
before file deletion. Read-time policy filtering remains active before cleanup.
Cleanup works with a draft or missing constitution and without a provider.

`test_character_retention.py` mirrors `test_retention_policy.py`: pre-call project
skips, pre/post model marker filtering, late-policy hiding, alias protections,
backup content, accepted-pair exclusion, lineage propagation, edited/deferred
queue coverage, snapshot races including a new judgment after backup, repeat application, and
independent memory roots.
Extend the existing retention tests for the combined backup and unchanged source
transcripts. Cleanup must work with a draft or missing constitution and no provider.

## Nightly integration

After curation, add two separately measured stages:

```text
dream-stage --continue-on-error character_generate -- dream character generate --source all --limit 10
dream-stage --continue-on-error character_judge -- dream character judge --limit 10
```

Both limits are operational values Szymon can adjust. With character disabled,
both stages report a clean skip and make no provider calls. Preflight is command-
scoped. The unit contains no literal provider commands. All provider selection
lives in config, including Claude-backed memory review and character judging.
Review, accept, edit, render, export, and eval never run in systemd.

Update `metrics.STAGES`, `finish_run`, `seed_success`, stage runner choices,
exporter/dashboard assumptions, and nightly-isolation tests for both stages.
Every failed stage preserves failed health while later stages still run.
Direct character commands return 1 if any item fails.

## Delivery order after approval

First write and review the implementation plan. It will divide work into small,
test-first commits. Start with M1 and the shared configuration, path protections,
versioned schema, and backup contract. Build retention support alongside every
table's introduction. Then add generation, judging, and human review. Add render,
export, and fixed-probe evaluation once accepted revision semantics are tested.
Nightly generation/judging and the disabled-by-default hook are included.

Before every implementation commit run:

```bash
pytest -q -p no:cacheprovider dream
git diff --check
```

Tests never invoke real Codex/Claude CLIs or use credentials/network. Provider
results and targets are fakes. No real transcript leaves the machine during
development. A live smoke test requires Szymon's explicit request. Do not push or
open a pull request unless asked.

## Approved decisions and repository gaps

Szymon approved these choices on 2026-10-18:

1. Judge runs nightly through config. Its family must differ from both extraction
   and regeneration families. Same-family override defaults to false.
2. Add `character_generate` and `character_judge` after curation, each with limit 10.
   Disabled character stages skip without provider calls.
3. Ship file rendering and separate trusted session-start hook injection. The
   injection flag defaults to false.
4. Render at most 3 pairs within 12000 total codepoints.
5. Constitution lives at `~/.config/dream/character/constitution.md`.
6. Use per-item model-failure isolation, command-scoped preflight, a version-3
   pre-DDL backup, and strict new-character config validation. Keep the existing
   review model defaults and memory suggestion cleanup unchanged.
7. Claude transcript source is reliable evidence of historical Anthropic overlap.
   Display it in review, export, and eval. Verdicts against chosen still need
   explicit human acknowledgement.
8. Seeds are optional for constitution approval. Seedless synthetic traits skip.
9. Retention deletes affected derived character records and lineage after backup.
   Character cleanup has one all-or-nothing snapshot check. User-modified managed
   files remain in place and are reported.

The earlier Claude-free nightly assumption was wrong. Local review already routes
through Claude. The binding rule is provider-neutral unit commands with routing
in config. Earlier recall docs also show old hook timeouts. Current code wins with
five-second hooks, `hookEventName`, query-aware rendering, and policy filtering.
The existing migration backup detector covers provider-column changes only;
character v3 adds a pre-DDL check. Existing config does not validate every old
stage value type; strict validation is limited to new character values. The existing
memory suggestion cleanup remains outside scope. No compatibility code is added.

## Remaining open decisions

None from design review. Stop only if planning finds a new conflict with code.

## Architecture consultation

Astra reviewed the draft architecture at Szymon's request. This revision addresses
its findings about historical model identity, stale-pair regeneration, constitution
retention checks, false-positive starvation, reconstructed task context, incomplete
eval evidence, and fixed-probe trait bindings. It also removes a redundant review
mode setting and limits deterministic security claims to host data flow.
Szymon subsequently approved the revised design and resolved its choices above.

## Implementation acceptance

Update and commit this approved spec, write and commit the implementation plan,
then implement M1–M8 test-first on `feat/character` without another plan review.
Before every commit run the full hermetic suite and `git diff --check`. Do not
run real providers against transcripts, edit local config, push, or open a PR.
Report commits, final test count, and anything deferred.
