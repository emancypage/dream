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


def test_selection_preserves_relevance_order_across_source_kinds_and_projects():
    values = [
        _candidate("best", "distilled_summary", "model_distilled", .9, project="z-project"),
        _candidate("second", "approved_memory", "user_approved", .5, project="a-project"),
    ]
    selected = select_recall_candidates(values, _query(), None, {})
    assert [candidate.document.id for candidate in selected] == ["best", "second"]


def test_selection_uses_the_threshold_of_the_active_mode():
    values = [_candidate("a", "approved_memory", "user_approved", .6), _candidate("b", "approved_memory", "user_approved", .3)]
    calibrations = {"lexical": {"threshold": .5}, "lexical_plus_embedder": {"threshold": .2}}
    assert [c.document.id for c in select_recall_candidates(values, _query(), None, calibrations)] == ["a"]
    assert [c.document.id for c in select_recall_candidates(values, _query(), None, calibrations, mode="lexical_plus_embedder")] == ["a", "b"]
    assert [c.document.id for c in select_recall_candidates(values, _query(), None, {"lexical": {"threshold": .5}}, mode="lexical_plus_embedder")] == ["a", "b"]


def test_calibration_lookup_reports_the_row_used():
    from recall_select import calibration_for
    calibrations = {"lexical": {"threshold": .5, "calibration_version": "lex-v1"}}
    assert calibration_for(calibrations, "lexical", _query()) == (.5, "lex-v1")
    assert calibration_for(calibrations, "lexical", _query("session-start:startup")) == (None, None)
    assert calibration_for(calibrations, "lexical_plus_embedder", _query()) == (None, None)
    assert calibration_for({}, "lexical", _query()) == (None, None)
