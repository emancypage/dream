"""Local UI adapter. All mutations use Dream's existing guarded review functions."""
from contextlib import closing
import sqlite3

from config import load_config
from storage import open_db_readonly
from pause_control import is_paused, PAUSE_NOTICE
from character import storage, review, short
from character.generate import prompt_version
from character.constitution import load_constitution
from character.safety import guard_path, check_constitution, policy_for


class ReviewApp:
    def __init__(self, config=None):
        self._config = config

    @property
    def config(self):
        return self._config or load_config()

    def _active(self):
        if is_paused():
            raise ValueError(PAUSE_NOTICE)

    def _connect(self):
        cfg = self.config
        guard_path(cfg.db_path, cfg)
        return open_db_readonly(cfg.db_path)

    @staticmethod
    def _pair_status(pair, document):
        status = pair['status']
        if status in ('pending', 'deferred', 'edited'):
            if pair['constitution_sha256'] != document.sha256:
                return 'stale'
            if pair.get('source_kind') == 'synthetic' or pair.get('prompt_version') != prompt_version():
                return 'needs_evidence'
        return status

    @staticmethod
    def _summary(row, kind, status=None):
        messages = row.get('prompt_messages', [])
        title = next((m.get('content', '') for m in reversed(messages)
                      if m.get('role') == 'user'), '')
        if kind == 'short':
            title = row['body']['situation']
        return dict(id=row['id'], kind=kind, status=status or row['status'],
                    hash=row['content_sha256'], title=title[:180],
                    created_at=row.get('created_at', ''), traits=row.get('trait_ids', []))

    def queue(self):
        self._active()
        cfg = self.config
        doc = load_constitution(cfg, require_approved=False)
        check_constitution(doc, policy_for(cfg))
        with closing(self._connect()) as conn:
            # One policy scan per collection; CLI list expands every record and
            # repeats full scans, which is too slow for an interactive queue.
            pairs = storage.eligible_records(conn, cfg, 'character_pairs')
            rows = storage.eligible_records(conn, cfg, short.TABLE)
            sources = {p['id']: p['content_sha256'] for p in review.accepted_pairs(conn, cfg, doc)}
            items = []
            for pair in pairs:
                status = self._pair_status(pair, doc)
                items.append(self._summary(pair, 'pair', status))
            examples = [self._summary(row, 'short', row['status'] if
                        row['constitution_sha256'] == doc.sha256 and
                        sources.get(row['pair_id']) == row['pair_sha256'] else 'stale') for row in rows]
        active = [p for p in items if p['status'] in ('pending', 'edited')]
        drafts = [p for p in examples if p['status'] == 'draft']
        history = [p for p in items + examples if p not in active and p not in drafts]
        key = lambda row: (row['created_at'], row['id'])
        return dict(pairs=sorted(active, key=key), shorts=sorted(drafts, key=key),
                    history=sorted(history, key=key, reverse=True),
                    counts=dict(pairs=len(active), shorts=len(drafts),
                                deferred=sum(p['status'] == 'deferred' for p in items),
                                stale_pairs=sum(p['status'] == 'stale' for p in items),
                                stale_shorts=sum(p['status'] == 'stale' for p in examples)))

    def detail(self, kind, id):
        self._active()
        cfg = self.config
        with closing(self._connect()) as conn:
            if kind == 'pair':
                row = review.show_pair(conn, cfg, id)
                doc = load_constitution(cfg, require_approved=False)
                row['status'] = self._pair_status(row, doc)
                current = [j for j in row['judgments'] if j.get('complete') is True
                           and j.get('constitution_sha256') == doc.sha256
                           and j.get('candidate_sha256') == review.candidate_hash(row)
                           and j.get('pair_revision') == row['revision']
                           and j.get('pair_sha256') == row['content_sha256']]
                can_prepare = False
                user_feedback = None
                if row.get('moment_id'):
                    try:
                        parent = review.check_source_parent(conn, cfg, row, doc)
                        user_feedback = parent.get('user_correction')
                    except ValueError:
                        pass
                if row['status'] == 'accepted' and cfg.data['character']['enabled']:
                    accepted = any(p['id'] == id for p in review.accepted_pairs(conn, cfg, doc))
                    existing = any(p['pair_id'] == id and p['pair_sha256'] == row['content_sha256']
                                   for p in storage.eligible_records(conn, cfg, short.TABLE))
                    can_prepare = accepted and not existing
                return dict(row, can_accept=bool(current) and row['status'] in ('pending', 'edited', 'deferred'),
                            can_prepare_short=can_prepare, current_judgments=current, user_feedback=user_feedback)
            if kind == 'short':
                row = short.show_example(conn, cfg, id)
                events = [e for e in storage.eligible_records(conn, cfg, 'character_review_events')
                          if e.get('short_id') == id]
                return dict(row, review_events=sorted(events, key=lambda e: e.get('created_at', '')))
        raise ValueError('Unknown review kind')

    def decide(self, kind, id, expected_sha, choice, reason, acknowledge_flags=False):
        self._active()
        cfg = self.config
        if kind == 'pair':
            if choice not in ('left', 'right', 'reject', 'defer'):
                raise ValueError('Unknown pair choice')
            action = 'accept' if choice in ('left', 'right') else choice
            with closing(self._connect()) as conn:
                if action == 'accept':
                    doc = load_constitution(cfg, require_approved=False)
                    pair = review.show_pair(conn, cfg, id)
                    if self._pair_status(pair, doc) == 'needs_evidence':
                        raise ValueError('current communication-problem evidence required')
                record = review.review_pair(conn, cfg, id, action, expected_sha=expected_sha,
                    reason=reason, answer=choice if action == 'accept' else 'left',
                    acknowledge_flags=acknowledge_flags)
            result = dict(saved=True, record=record)
            if action == 'accept':
                # Reopen after the commit, as the CLI does. A provider failure
                # must never hide or undo a successfully recorded human choice.
                try:
                    self._active()
                    with closing(self._connect()) as conn:
                        result['short_draft'] = short.generate_example(conn, cfg, id,
                            expected_pair_sha=record['content_sha256'])
                except (ValueError, OSError, sqlite3.Error) as exc:
                    result['short_draft'] = short.draft_failure(id, record['content_sha256'], exc)
            return result
        if kind == 'short':
            with closing(self._connect()) as conn:
                record = short.review_example(conn, cfg, id, choice, expected_sha=expected_sha, reason=reason)
            return dict(saved=True, record=record)
        raise ValueError('Unknown review kind')

    def edit_short(self, id, expected_sha, expected_pair_sha, body):
        self._active()
        cfg = self.config
        with closing(self._connect()) as conn:
            current = short.show_example(conn, cfg, id)
            if current['content_sha256'] != expected_sha or current['status'] == 'stale':
                raise ValueError('short example changed or unavailable')
            record = short.draft_example(conn, cfg, current['pair_id'],
                expected_pair_sha=expected_pair_sha, expected_sha=expected_sha, body=body)
        return dict(saved=True, record=record)

    def delete_pair(self, id, expected_sha, reason=''):
        self._active()
        with closing(self._connect()) as conn:
            return review.delete_pair(conn, self.config, id, expected_sha=expected_sha, reason=reason)

    def prepare_short(self, id, expected_sha):
        self._active()
        cfg = self.config
        with closing(self._connect()) as conn:
            result = short.generate_example(conn, cfg, id, expected_pair_sha=expected_sha)
        return dict(short_draft=result)
