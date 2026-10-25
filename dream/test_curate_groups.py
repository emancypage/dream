"""Grouped review contracts, using temporary memory and fake providers only."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

import curate
import dream
from test_curate import (_decision, _curation_payload, _fixture_snapshot,
                         _insert_pending, make_curation_fixture, run_curate, statuses)


def stage(db, root, path, body, kind="update"):
    conn = dream.open_db(db)
    result = _insert_pending(conn, root, path, kind, body, preview=False)
    conn.commit()
    conn.close()
    return result


def empty_env(tmp_path):
    root, db, config = make_curation_fixture(tmp_path, target="seed.md")
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM suggestions")
    conn.commit()
    conn.close()
    return root, db, config


def fake_provider(monkeypatch, review):
    def generate(stage, prompt, schema, **kwargs):
        assert stage == "review"
        return SimpleNamespace(output={"decisions": review(_curation_payload(prompt))},
                               provider="fake", model="local-test")
    monkeypatch.setattr(dream, "generate", generate, raising=False)


@pytest.mark.parametrize("as_text", [False, True])
@pytest.mark.parametrize(("field", "size", "message"), [
    ("reason", 2001, "reason.*2000"),
    ("body", 60001, "body.*60000"),
])
def test_review_output_bounds_are_host_enforced(as_text, field, size, message):
    decision = _decision(7, "merge", **{field: "😀" * size})
    output = {"decisions": [decision]}
    if as_text:
        output = json.dumps(output, ensure_ascii=False)
    with pytest.raises(ValueError, match=message):
        curate.parse_curation_output(output, {7})


@pytest.mark.parametrize("as_text", [False, True])
def test_total_output_bound_includes_dictionary_results(as_text):
    output = {"decisions": [_decision(i, "merge", body="😀" * 60000) for i in range(4)]}
    if as_text:
        output = json.dumps(output, ensure_ascii=False)
    with pytest.raises(ValueError, match="output.*240000"):
        curate.parse_curation_output(output, range(4))


def test_output_field_bounds_accept_exact_unicode_limits():
    output = {"decisions": [_decision(1, "merge", reason="😀" * 2000, body="😀" * 60000)]}
    result = curate.parse_curation_output(output, {1})
    assert len(result[1].reason) == 2000
    assert len(result[1].body) == 60000


def test_groups_resolved_aliases_together_and_reviews_index_last(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "topic.md").write_text("original topic", encoding="utf-8")
    (root / "other.md").write_text("original other", encoding="utf-8")
    index = stage(db, root, "MEMORY.md", "- [Topic](topic.md) final description\n", "index")
    ids = [stage(db, root, path, f"proposal {i}") for i, path in enumerate(
        ["topic.md", "./topic.md", "sub/../topic.md"])]
    other = stage(db, root, "other.md", "final other")
    groups = []
    def review(payload):
        rows = payload["suggestions"]
        group_ids = [row["id"] for row in rows]
        groups.append(group_ids)
        if group_ids == ids:
            assert len(payload["targets"]) == 1
            assert {row["target_ref"] for row in rows} == {"topic.md"}
            assert [row["target_path"] for row in rows] == ["topic.md", "./topic.md", "sub/../topic.md"]
            return [_decision(ids[0], "merge", body="all topic facts"),
                    *[_decision(i, "reject", "incorporated in final topic") for i in ids[1:]]]
        if group_ids == [index]:
            context = payload["topic_context"]
            assert context[0]["current_target_body"] == "all topic facts"
            assert context[0]["outcome"] == "applied"
        return [_decision(i, "accept") for i in group_ids]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert groups == [ids, [other], [index]]
    assert (root / "topic.md").read_text() == "all topic facts"
    assert "topic.md" in (root / "MEMORY.md").read_text()
    assert statuses(db) == {index: "accepted", ids[0]: "accepted", ids[1]: "rejected",
                            ids[2]: "rejected", other: "accepted"}


@pytest.mark.parametrize("failure", ["provider", "oversized", "schema"])
def test_failed_group_keeps_other_progress_and_prints_partial_summary(tmp_path, monkeypatch, capsys, failure):
    root, db, config = empty_env(tmp_path)
    bad = stage(db, root, "bad.md", "x" * 60001 if failure == "oversized" else "bad topic", "new")
    good = stage(db, root, "good.md", "good topic", "new")
    calls = []
    def review(payload):
        ids = [row["id"] for row in payload["suggestions"]]
        calls.extend(ids)
        if ids == [bad]:
            if failure == "provider":
                raise RuntimeError("private provider content must not leak")
            return []
        return [_decision(good, "accept")]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 1
    assert statuses(db) == {bad: "pending", good: "accepted"}
    assert not (root / "bad.md").exists()
    assert (root / "good.md").read_text() == "good topic"
    out = capsys.readouterr()
    assert f"applied IDs: [{good}]" in out.out
    assert f"unresolved IDs: [{bad}]" in out.out
    assert "private provider content" not in out.out + out.err
    if failure == "oversized":
        assert bad not in calls


def test_competing_topic_writes_invalidate_all_decisions(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "topic.md").write_text("keep original")
    ids = [stage(db, root, "topic.md", text) for text in ("first fact", "second fact", "third fact")]
    before = _fixture_snapshot(root, db)
    fake_provider(monkeypatch, lambda _: [_decision(ids[0], "accept"), _decision(ids[1], "accept"),
                                          _decision(ids[2], "reject", "incorporated")])
    assert run_curate(root, db, config) == 1
    assert _fixture_snapshot(root, db) == before
    assert statuses(db) == {i: "pending" for i in ids}


def test_combined_write_conflict_keeps_incorporated_siblings_pending(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    target = root / "topic.md"
    target.write_text("original")
    ids = [stage(db, root, "topic.md", text) for text in ("first", "second")]
    def review(_):
        target.write_text("external edit")
        return [_decision(ids[0], "merge", body="combined"), _decision(ids[1], "reject", "incorporated")]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert target.read_text() == "external edit"
    assert statuses(db) == {i: "pending" for i in ids}


def test_resolved_identity_change_with_identical_content_defers_group(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    for name in ("a.md", "b.md"):
        (root / name).write_text("same original")
    alias = root / "alias.md"
    alias.symlink_to("a.md")
    item = stage(db, root, "alias.md", "new content")
    def review(_):
        alias.unlink()
        alias.symlink_to("b.md")
        return [_decision(item, "accept")]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert (root / "b.md").read_text() == "same original"
    assert statuses(db)[item] == "pending"


def test_concurrently_resolved_sibling_prevents_combined_apply(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "topic.md").write_text("original")
    ids = [stage(db, root, "topic.md", text) for text in ("first", "second")]
    def review(_):
        other = sqlite3.connect(db)
        other.execute("UPDATE suggestions SET status='accepted' WHERE id=?", (ids[1],))
        other.commit()
        other.close()
        return [_decision(ids[0], "merge", body="combined"), _decision(ids[1], "reject", "incorporated")]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert (root / "topic.md").read_text() == "original"
    assert statuses(db) == {ids[0]: "pending", ids[1]: "accepted"}


def test_failed_new_topic_keeps_index_pending_then_retry_completes(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    index = stage(db, root, "MEMORY.md", "- [New](new.md) final topic\n", "index")
    topic = stage(db, root, "new.md", "final topic", "new")
    state = {"fail": True}
    def review(payload):
        ids = [row["id"] for row in payload["suggestions"]]
        if ids == [topic] and state["fail"]:
            raise RuntimeError("provider failed")
        return [_decision(i, "accept") for i in ids]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 1
    assert statuses(db) == {index: "pending", topic: "pending"}
    assert not (root / "MEMORY.md").exists()
    state["fail"] = False
    assert run_curate(root, db, config) == 0
    assert statuses(db) == {index: "accepted", topic: "accepted"}
    assert "new.md" in (root / "MEMORY.md").read_text()


def test_mixed_ready_and_pending_index_additions_defer_entire_proposal(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "ready.md").write_text("ready topic")
    (root / "waiting.md").write_text("old topic")
    topic = stage(db, root, "waiting.md", "needed new topic")
    index = stage(db, root, "MEMORY.md", "- [Ready](ready.md) ready\n- [Waiting](waiting.md) new facts\n", "index")
    fake_provider(monkeypatch, lambda data: [_decision(row["id"], "defer" if row["id"] == topic else "accept")
                                            for row in data["suggestions"]])
    assert run_curate(root, db, config) == 0
    assert statuses(db) == {topic: "pending", index: "pending"}
    assert not (root / "MEMORY.md").exists()


@pytest.mark.parametrize("changed", ["topic", "index"])
def test_index_dependency_and_index_races_defer_the_index(tmp_path, monkeypatch, changed):
    root, db, config = empty_env(tmp_path)
    (root / "topic.md").write_text("final topic")
    (root / "MEMORY.md").write_text("# Index\n")
    index = stage(db, root, "MEMORY.md", "- [Topic](topic.md) final description\n", "index")
    def review(data):
        assert data["topic_context"][0]["current_target_body"] == "final topic"
        (root / ("topic.md" if changed == "topic" else "MEMORY.md")).write_text("external edit")
        return [_decision(index, "accept")]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert statuses(db)[index] == "pending"
    assert "topic.md" not in (root / "MEMORY.md").read_text()


def test_new_index_reference_without_reviewed_context_is_deferred(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "a.md").write_text("a")
    (root / "b.md").write_text("b")
    index = stage(db, root, "MEMORY.md", "- [A](a.md) a\n", "index")
    fake_provider(monkeypatch, lambda _: [_decision(index, "merge", body="- [B](b.md) invented description\n")])
    assert run_curate(root, db, config) == 0
    assert statuses(db)[index] == "pending"


def test_grouped_dry_run_makes_no_writes_and_labels_prospective_outcomes(tmp_path, monkeypatch, capsys):
    root, db, config = empty_env(tmp_path)
    topic = stage(db, root, "topic.md", "final topic", "new")
    stage(db, root, "MEMORY.md", "- [Topic](topic.md) final\n", "index")
    before = _fixture_snapshot(root, db)
    fake_provider(monkeypatch, lambda data: [_decision(row["id"], "accept") for row in data["suggestions"]])
    assert run_curate(root, db, config, dry_run=True) == 0
    assert _fixture_snapshot(root, db) == before
    assert not list(tmp_path.glob("memory-backups/*"))
    out = capsys.readouterr().out
    assert "prospective" in out
    assert "applied IDs: []" in out
    assert f"{topic}" in out


def test_oversized_topic_group_is_not_split_to_make_it_fit(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    bad_ids = [stage(db, root, "large.md", "x" * 59999, "new") for _ in range(5)]
    good = stage(db, root, "good.md", "good", "new")
    seen = []
    def review(data):
        seen.extend(row["id"] for row in data["suggestions"])
        return [_decision(good, "accept")]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 1
    assert seen == [good]
    assert all(statuses(db)[i] == "pending" for i in bad_ids)
    assert statuses(db)[good] == "accepted"


def test_index_context_overrun_keeps_topics_successful_and_index_pending(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    good = stage(db, root, "good.md", "good", "new")
    additions = []
    for i in range(6):
        name = f"large-{i}.md"
        (root / name).write_text("z" * 45000)
        additions.append(f"- [Large]({name}) facts")
    index = stage(db, root, "MEMORY.md", "\n".join(additions), "index")
    seen = []
    def review(data):
        seen.extend(row["id"] for row in data["suggestions"])
        return [_decision(good, "accept")]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 1
    assert seen == [good]
    assert statuses(db) == {good: "accepted", index: "pending"}
    assert (root / "good.md").read_text() == "good"


def test_merge_cannot_drop_unresolved_original_index_additions(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "ready.md").write_text("ready")
    index = stage(db, root, "MEMORY.md", "- [Ready](ready.md) ready\n- [Missing](missing.md) waiting\n", "index")
    fake_provider(monkeypatch, lambda _: [_decision(index, "merge", body="- [Ready](ready.md) ready\n")])
    assert run_curate(root, db, config) == 0
    assert statuses(db)[index] == "pending"
    assert not (root / "MEMORY.md").exists()


def test_new_pending_topic_work_during_index_review_defers_addition(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "topic.md").write_text("reviewed topic")
    index = stage(db, root, "MEMORY.md", "- [Topic](topic.md) reviewed facts\n", "index")
    def review(_):
        stage(db, root, "topic.md", "new pending facts")
        return [_decision(index, "accept")]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert statuses(db)[index] == "pending"
    assert not (root / "MEMORY.md").exists()


def test_new_sibling_during_topic_review_defers_whole_group(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "topic.md").write_text("original")
    first = stage(db, root, "topic.md", "first facts")
    def review(_):
        stage(db, root, "topic.md", "new sibling")
        return [_decision(first, "accept")]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert set(statuses(db).values()) == {"pending"}
    assert (root / "topic.md").read_text() == "original"


def test_failed_combined_write_keeps_sibling_rejections_and_original_patch_retryable(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "topic.md").write_text("original")
    ids = [stage(db, root, "topic.md", body) for body in ("first", "second")]
    fake_provider(monkeypatch, lambda _: [_decision(ids[0], "merge", body="combined"),
                                          _decision(ids[1], "reject", "incorporated")])
    def fail(*a, **kw):
        raise OSError("private failure detail")
    monkeypatch.setattr(dream, "_apply_suggestion", fail)
    assert run_curate(root, db, config) == 1
    assert statuses(db) == {i: "pending" for i in ids}
    assert (root / "topic.md").read_text() == "original"


def test_group_finalization_happens_once_after_index(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "MEMORY.md").write_text("# Index\n")
    index = stage(db, root, "MEMORY.md", "- [Topic](topic.md) facts\n", "index")
    topic = stage(db, root, "topic.md", "facts", "new")
    events = []
    real_prune, real_sync = dream._prune_orphaned_index_lines, dream._sync_recall_documents
    def review(data):
        ids = [row["id"] for row in data["suggestions"]]
        events.append(ids)
        return [_decision(i, "accept") for i in ids]
    fake_provider(monkeypatch, review)
    def prune(*a, **kw):
        events.append("prune")
        return real_prune(*a, **kw)
    def sync(*a, **kw):
        events.append("sync")
        return real_sync(*a, **kw)
    monkeypatch.setattr(dream, "_prune_orphaned_index_lines", prune)
    monkeypatch.setattr(dream, "_sync_recall_documents", sync)
    assert run_curate(root, db, config) == 0
    assert events == [[topic], [index], "prune", "sync"]
    assert len(list(tmp_path.glob("memory-backups/*"))) == 1


def test_resolved_memory_index_stays_append_only_for_update_proposal(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "old.md").write_text("old facts")
    (root / "new.md").write_text("new facts")
    (root / "MEMORY.md").write_text("- [Old](old.md) keep\n")
    item = stage(db, root, "./MEMORY.md", "- [New](new.md) add\n", "update")
    fake_provider(monkeypatch, lambda _: [_decision(item, "accept")])
    assert run_curate(root, db, config) == 0
    assert (root / "MEMORY.md").read_text() == "- [Old](old.md) keep\n- [New](new.md) add\n"


def test_dependency_new_retention_exclusion_during_review_defers_index(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "topic.md").write_text("private topic facts")
    item = stage(db, root, "MEMORY.md", "- [Topic](topic.md) facts\n", "index")
    def review(_):
        (root / ".dream-retention.json").write_text('{"text_markers":["private topic"]}')
        return [_decision(item, "accept")]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert statuses(db)[item] == "pending"
    assert not (root / "MEMORY.md").exists()


def test_final_recall_sync_failure_returns_nonzero_with_applied_ids(tmp_path, monkeypatch, capsys):
    root, db, config = empty_env(tmp_path)
    item = stage(db, root, "topic.md", "facts", "new")
    fake_provider(monkeypatch, lambda _: [_decision(item, "accept")])
    monkeypatch.setattr(dream, "_sync_recall_documents", lambda *a, **kw: False)
    assert run_curate(root, db, config) == 1
    assert statuses(db)[item] == "accepted"
    assert f"applied IDs: [{item}]" in capsys.readouterr().out


def test_unresolvable_target_does_not_block_an_independent_group(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "cycle").symlink_to("cycle")
    bad = stage(db, root, "cycle/topic.md", "unreadable", "new")
    good = stage(db, root, "good.md", "good", "new")
    fake_provider(monkeypatch, lambda _: [_decision(good, "accept")])
    assert run_curate(root, db, config) == 1
    assert statuses(db) == {bad: "pending", good: "accepted"}
    assert (root / "good.md").read_text() == "good"


@pytest.mark.parametrize("excluded_field", ["path", "proposal", "current_body", "provenance"])
def test_retention_refresh_prevents_later_group_content_from_reaching_provider(tmp_path, monkeypatch, excluded_field):
    root, db, config = empty_env(tmp_path)
    first = stage(db, root, "first.md", "first facts", "new")
    (root / "private.md").write_text("private old body")
    private = stage(db, root, "private.md", "private new proposal")
    if excluded_field == "provenance":
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO sessions(session_id,project_slug,jsonl_path,jsonl_mtime) VALUES ('session-a','private-project','/fake',1)")
        conn.execute("UPDATE suggestions SET source_sessions='' WHERE id=?", (first,))
        conn.commit()
        conn.close()
    seen = []
    def review(data):
        ids = [row["id"] for row in data["suggestions"]]
        seen.extend(ids)
        if ids == [first]:
            policy = {"path": {"memory_paths": ["private.md"]},
                      "proposal": {"text_markers": ["private new proposal"]},
                      "current_body": {"text_markers": ["private old body"]},
                      "provenance": {"project_slugs": ["private-project"]}}[excluded_field]
            (root / ".dream-retention.json").write_text(json.dumps(policy))
            return [_decision(first, "accept")]
        pytest.fail("newly excluded content reached the provider")
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert seen == [first]
    assert statuses(db) == {first: "accepted", private: "pending"}


def test_unresolvable_pending_path_does_not_block_ready_index_context(tmp_path, monkeypatch):
    root, db, config = empty_env(tmp_path)
    (root / "cycle").symlink_to("cycle")
    bad = stage(db, root, "cycle/topic.md", "unreadable", "new")
    (root / "good.md").write_text("good topic")
    index = stage(db, root, "MEMORY.md", "- [Good](good.md) facts\n", "index")
    fake_provider(monkeypatch, lambda _: [_decision(index, "accept")])
    assert run_curate(root, db, config) == 1
    assert statuses(db) == {bad: "pending", index: "accepted"}
    assert "good.md" in (root / "MEMORY.md").read_text()



def existing_index_env(tmp_path):
    root, db, config = empty_env(tmp_path)
    (root / 'topic.md').write_text('Current topic facts')
    (root / 'keep.md').write_text('Keep facts')
    (root / 'MEMORY.md').write_text('# Topics\n- [Topic](topic.md) stale description\n- [Keep](keep.md) keep description\n')
    item = stage(db, root, 'MEMORY.md', '- [Topic](topic.md) current description\n', 'index')
    return root, db, config, item


def test_existing_description_update_has_topic_context_and_preserves_omitted_entries(tmp_path, monkeypatch):
    root, db, config, item = existing_index_env(tmp_path)
    def review(data):
        assert data['topic_context'][0]['target_path'] == 'topic.md'
        assert data['topic_context'][0]['current_target_body'] == 'Current topic facts'
        assert data['topic_context'][0]['ready'] is True
        return [_decision(item, 'accept')]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert statuses(db)[item] == 'accepted'
    assert (root / 'MEMORY.md').read_text() == '# Topics\n- [Topic](topic.md) current description\n- [Keep](keep.md) keep description\n'


@pytest.mark.parametrize('change', ['topic', 'index', 'pending', 'retention'])
def test_existing_description_update_defers_on_dependency_or_index_change(tmp_path, monkeypatch, change):
    root, db, config, item = existing_index_env(tmp_path)
    before = (root / 'MEMORY.md').read_text()
    def review(data):
        assert data['topic_context'][0]['current_target_body'] == 'Current topic facts'
        if change == 'topic':
            (root / 'topic.md').write_text('External topic change')
        elif change == 'index':
            (root / 'MEMORY.md').write_text('External index change')
        elif change == 'pending':
            stage(db, root, 'topic.md', 'Pending update')
        else:
            (root / '.dream-retention.json').write_text('{"memory_paths":["topic.md"]}')
        return [_decision(item, 'accept')]
    fake_provider(monkeypatch, review)
    assert run_curate(root, db, config) == 0
    assert statuses(db)[item] == 'pending'
    assert (root / 'MEMORY.md').read_text() == ('External index change' if change == 'index' else before)


def test_existing_description_with_deferred_topic_work_stays_pending(tmp_path, monkeypatch):
    root, db, config, item = existing_index_env(tmp_path)
    topic = stage(db, root, 'topic.md', 'Pending new topic facts')
    fake_provider(monkeypatch, lambda data: [_decision(row['id'], 'defer' if row['id'] == topic else 'accept') for row in data['suggestions']])
    assert run_curate(root, db, config) == 0
    assert statuses(db) == {item: 'pending', topic: 'pending'}
    assert 'stale description' in (root / 'MEMORY.md').read_text()


def test_conflicting_index_decisions_apply_no_rows_or_descriptions(tmp_path, monkeypatch):
    root, db, config, item = existing_index_env(tmp_path)
    other = stage(db, root, './MEMORY.md', '- [Topic](topic.md) conflicting description\n', 'index')
    fake_provider(monkeypatch, lambda _: [_decision(item, 'accept'), _decision(other, 'accept')])
    before = _fixture_snapshot(root, db)
    assert run_curate(root, db, config) == 1
    assert _fixture_snapshot(root, db) == before
    assert statuses(db) == {item: 'pending', other: 'pending'}


@pytest.mark.parametrize('late_change', ['index', 'proposal', 'status'])
def test_index_final_writer_rechecks_index_and_reviewed_pending_proposal(tmp_path, monkeypatch, late_change):
    root, db, config, item = existing_index_env(tmp_path)
    original = (root / 'MEMORY.md').read_text()
    fake_provider(monkeypatch, lambda _: [_decision(item, 'accept')])
    real_merge = dream._merge_index
    writer_merges = []
    def merge(existing, proposed):
        output = real_merge(existing, proposed)
        # The final writer merges a proposal only after opening its write connection.
        if writer_merges:
            if late_change == 'index':
                (root / 'MEMORY.md').write_text('Late external index edit')
            elif late_change == 'proposal':
                writer_merges[0].execute('UPDATE suggestions SET body=? WHERE id=?', ('Changed proposal', item))
            else:
                writer_merges[0].execute("UPDATE suggestions SET status='rejected' WHERE id=?", (item,))
        return output
    real_apply = dream._apply_suggestion
    def apply(conn, *args, **kwargs):
        writer_merges.append(conn)
        try:
            return real_apply(conn, *args, **kwargs)
        finally:
            writer_merges.clear()
    monkeypatch.setattr(dream, '_merge_index', merge)
    monkeypatch.setattr(dream, '_apply_suggestion', apply)
    assert run_curate(root, db, config) == 0
    assert statuses(db)[item] != 'accepted'
    assert (root / 'MEMORY.md').read_text() == ('Late external index edit' if late_change == 'index' else original)


def test_accepted_original_description_cannot_undo_another_accepted_update(tmp_path, monkeypatch):
    root, db, config, item = existing_index_env(tmp_path)
    old = stage(db, root, './MEMORY.md', '- [Topic](topic.md) stale description\n', 'index')
    fake_provider(monkeypatch, lambda _: [_decision(item, 'accept'), _decision(old, 'accept')])
    before = _fixture_snapshot(root, db)
    assert run_curate(root, db, config) == 1
    assert _fixture_snapshot(root, db) == before
    assert statuses(db) == {item: 'pending', old: 'pending'}


def test_identical_accepted_index_descriptions_share_one_result(tmp_path, monkeypatch):
    root, db, config, item = existing_index_env(tmp_path)
    second = stage(db, root, './MEMORY.md', '- [Topic](topic.md) current description\n', 'index')
    fake_provider(monkeypatch, lambda _: [_decision(item, 'accept'), _decision(second, 'accept')])
    assert run_curate(root, db, config) == 0
    assert statuses(db) == {item: 'accepted', second: 'accepted'}
    assert (root / 'MEMORY.md').read_text().count('(topic.md)') == 1
    assert 'current description' in (root / 'MEMORY.md').read_text()


@pytest.mark.parametrize(('field', 'value'), [('kind', 'new'), ('target_path', './MEMORY.md')])
def test_index_final_writer_requires_exact_original_reviewed_fields(tmp_path, monkeypatch, field, value):
    root, db, config, item = existing_index_env(tmp_path)
    before = (root / 'MEMORY.md').read_bytes()
    fake_provider(monkeypatch, lambda _: [_decision(item, 'accept')])
    real_apply = dream._apply_suggestion
    def apply(conn, *args, **kwargs):
        conn.execute(f'UPDATE suggestions SET {field}=? WHERE id=?', (value, item))
        return real_apply(conn, *args, **kwargs)
    monkeypatch.setattr(dream, '_apply_suggestion', apply)
    assert run_curate(root, db, config) == 0
    assert statuses(db)[item] == 'pending'
    assert (root / 'MEMORY.md').read_bytes() == before
