import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from curate import (  # noqa: E402
    CURATION_SCHEMA,
    CurationDecision,
    build_curation_prompt,
    parse_curation_output,
)
from model_types import SchemaValidationError, validate_output  # noqa: E402


def _decision(suggestion_id, decision="defer", reason="needs more evidence", **extra):
    value = {
        "suggestion_id": suggestion_id,
        "decision": decision,
        "reason": reason,
    }
    value.update(extra)
    return value


def test_parse_curation_output_returns_typed_decisions():
    result = parse_curation_output(
        {"decisions": [_decision(7, "merge", "keeps newer facts", body="merged")]},
        {7},
    )

    assert result == {
        7: CurationDecision(
            suggestion_id=7,
            decision="merge",
            reason="keeps newer facts",
            body="merged",
        )
    }


@pytest.mark.parametrize(
    ("output", "expected_ids", "message"),
    [
        ({"decisions": [_decision(7), _decision(7)]}, {7}, "duplicate"),
        ({"decisions": [_decision(8)]}, {7}, "suggestion IDs"),
        ({"decisions": [_decision(7), _decision(8)]}, {7}, "suggestion IDs"),
    ],
)
def test_parse_curation_output_rejects_duplicate_missing_and_unknown_ids(
    output, expected_ids, message
):
    with pytest.raises(ValueError, match=message):
        parse_curation_output(output, expected_ids)


def test_parse_curation_output_rejects_body_for_non_merge_decision():
    with pytest.raises(ValueError, match="body.*accept"):
        parse_curation_output(
            {"decisions": [_decision(7, "accept", "matches current facts", body="wrong")]},
            {7},
        )


def test_curation_schema_rejects_extra_keys_at_both_object_levels():
    valid = {"decisions": [_decision(7)]}

    with pytest.raises(SchemaValidationError, match="unexpected keys: extra"):
        validate_output({**valid, "extra": True}, CURATION_SCHEMA)
    with pytest.raises(SchemaValidationError, match="unexpected keys: extra"):
        validate_output(
            {"decisions": [{**_decision(7), "extra": True}]}, CURATION_SCHEMA
        )


def test_curation_schema_requires_non_empty_reason():
    with pytest.raises(ValueError, match="reason.*non-empty"):
        parse_curation_output(
            {"decisions": [_decision(7, reason="   ")]},
            {7},
        )


def test_build_curation_prompt_contains_bounded_review_context_and_safety_rules():
    prompt = build_curation_prompt(
        [
            {
                "id": 7,
                "kind": "update",
                "target_path": "topics/python.md",
                "body": "proposal body",
                "rationale": "new evidence",
                "source_sessions": ["session-a", "session-b"],
                "current_body": "live body",
                "current_sha256": "live-sha",
                "base_sha256": "base-sha",
                "target_existed": True,
                "conflict": True,
            }
        ],
        Path("/memory/root"),
    )

    for expected in (
        "7",
        "update",
        "topics/python.md",
        "proposal body",
        "new evidence",
        "session-a",
        "session-b",
        "live body",
        "live-sha",
        "base-sha",
        '"target_existed": true',
        '"conflict": true',
    ):
        assert expected in prompt
    assert "never invent" in prompt
    assert "MEMORY.md" in prompt
    assert "only intended index lines" in prompt
    assert "complete replacement body" in prompt
    assert "accept, reject, merge, defer" in prompt
    assert "concise" in prompt
