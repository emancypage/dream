# Dream Review

A local Codex MCP App for reviewing Dream character examples in Polish.

Requires Python 3.11+ and an installed `dream` command on the local host's PATH.
There are no additional runtime dependencies. The plugin resolves that
executable to Dream's Python source and uses its configured paths and providers.

## Use

Open **Dream Review** from the desktop app's left sidebar, from the chat's
plugin views, or ask to open it beside the chat. Review batches of five full
pairs, select A or B (or reject/defer), add a
reason, and save. Acceptance also requests a short draft using Dream's configured
provider. It remains unapproved until the separate short review.

Use **Usuń parę na zawsze** to delete a pair without choosing an answer or giving
a reason. It removes matching pairs, their judgments, short examples and review
decisions from the active database. Stable hashes prevent the
same prompt or source from returning, even after refresh or configuration changes.
Deletion is also available in History and for stale pairs.

The optional **Powód usunięcia** field saves the human reason plus a subject
excerpt (up to 500 characters) and project identity, without retaining either
answer. Reasons are passed to memory consolidation and staleness audits as
retention evidence. New feedback triggers an audit on the next subset
consolidation even without new sessions. A successful audit records the
feedback as considered; provider failures leave it pending for retry. The
feedback also remains context for later consolidation and audits. Suggestions
that cite a deletion-feedback ID carry its original human reason into the
configured review stage. The existing suggestion review and application rules
still decide whether a memory change is applied. Not wanting to assess a pair
does not itself mean the human wants its topic erased from memory. Existing database
backups and previously exported files are not erased.

Short examples show all five fields. Editing saves a new draft. Approving that
exact draft is a separate action. History includes prior choices, deferred
pairs, and stale examples. Stale examples cannot be accepted or edited here;
their source must first be prepared again with the existing Dream workflow.

Unfinished input is saved locally in the panel's browser storage, keyed by the
record ID and hash. Completed decisions are written to Dream's real database.
The panel never treats a model judgment as a human decision. Its data and write
tools have MCP Apps `app` visibility; only `show_review` is exposed to the model.
Writes also require the token issued with the exact displayed record hash.

The backend preserves Dream's pause, retention, path, source, version,
transaction, and backup checks. The recall MCP server stays read-only. The
opening and data-read tools make no provider calls; accepting a pair can make
one call to the configured character generation provider. A failed short draft
does not undo the saved human choice. Check History before retrying a timed-out
save, because the choice may already have been recorded.

## Pair relevance

The nightly character stage generates from actual user corrections only.
Extraction requires a concrete communication problem visible in the response
and user feedback; ordinary task changes and ambiguous feedback are skipped.
The panel shows the original user feedback when its source hash is current.
Unreviewed synthetic pairs and pairs from an earlier extraction prompt version
are kept in History as requiring evidence, rather than offered in the active
queue. Accepted pairs and approved short examples keep their human decisions.
The quality filter is model-assisted, so its judgments remain subject to human
review.

## Local installation

Register this directory as a local source in your Codex marketplace, then run:

```sh
codex plugin add dream-review@personal
```

The source entry uses `name: dream-review` and a local path to this directory.
Use the installed CLI's `codex plugin marketplace --help` for marketplace setup.
The portable manifests are `plugin.json` and `mcp.json`. Installation needs no
account upload or hosted service.

## Checks

From the Dream repository root:

```sh
python -m pytest dream/test_review_plugin.py
```

These tests use isolated fixture databases and fake providers. Never test write
controls against the user's real review queue without an explicit decision for
the displayed example.
