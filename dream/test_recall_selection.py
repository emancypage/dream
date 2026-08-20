from recall_select import select_recall_candidates
from recall_types import RecallCandidate, RecallDocument, RecallQuery, RecallSettings, AdapterSettings


def _candidate(doc_id, kind, trust, score, sha=None, project=None):
    doc = RecallDocument(doc_id, sha or doc_id, kind, trust, project, f"{doc_id}.md", "2026-08-20T00:00:00Z", "2026-08-20T00:00:00Z", "v1", doc_id)
    return RecallCandidate(doc, doc_id, {"fts": 1}, score, 1)


def _query(event="prompt"):
    return RecallQuery("query", "s", event, None, (), 4000, frozenset(), False)


def test_prompt_threshold_and_exclusions_apply():
    values = [_candidate("a", "approved_memory", "user_approved", .9), _candidate("b", "approved_memory", "user_approved", .1), _candidate("c", "raw_transcript", "untrusted_transcript", .9)]
    selected = select_recall_candidates(values, _query(), RecallSettings(True, False, False, True, 6000, 4000, 1800, 1200, "d", "c", AdapterSettings(False, "none", False), AdapterSettings(False, "none", False)), {"lexical": {"threshold": .5}})
    assert [candidate.document.id for candidate in selected] == ["a"]


def test_session_start_has_no_threshold_and_deduplicates_hashes():
    values = [_candidate("a", "approved_memory", "user_approved", .01, "same"), _candidate("b", "distilled_summary", "model_distilled", .9, "same")]
    selected = select_recall_candidates(values, _query("session-start:startup"), None, {})
    assert len(selected) == 1
