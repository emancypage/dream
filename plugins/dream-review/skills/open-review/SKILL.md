---
name: open-review
description: Open the local Dream Review panel when the user wants to compare character answer pairs, record human choices, or review short examples.
---

# Dream Review

Call `show_review` to open the local panel beside the chat.

The human chooses answers and approves exact short wording in the panel.
Do not choose, click acceptance controls, or approve examples on the user's
behalf. An instruction to install or test the plugin is not a review decision.
Use synthetic fixtures for write tests. Reading the live queue is allowed.

The human may also permanently delete a pair without selecting an answer or
giving a reason. An optional deletion reason and short subject context inform
future Dream memory curation; distinguish not wanting to rate an example from
an explicit request to forget a topic. Never invent a deletion reason or act
on a pair on the human's behalf.

The panel writes through Dream's existing character review functions. It
preserves exact hashes, human reasons, conflict acknowledgement, retention,
pause state, backups, and separate short approval. It never edits the
constitution. Stale examples need fresh preparation before review.

If Dream reports paused, stop reading its data and keep working with ordinary
tools. Resume only when the user explicitly asks.
