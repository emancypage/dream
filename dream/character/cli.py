"""Provider-neutral character commands with explicit read and write boundaries."""
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys

from storage import open_db_readonly
from character.types import canonical_json
from character.safety import guard_path, write_artifacts


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def register(subparsers):
    parser = subparsers.add_parser('character', help='Generate and review human-approved character examples')
    commands = parser.add_subparsers(dest='character_cmd', required=True)
    for name in ('estimate', 'generate'):
        command = commands.add_parser(name)
        command.add_argument('--source', choices=('corrections', 'synthetic', 'all'), default='all')
        command.add_argument('--limit', type=positive, required=True)
        command.add_argument('--refresh', action='store_true')
        command.add_argument('--expand-seeds', action='store_true')
    command = commands.add_parser('judge')
    command.add_argument('--limit', type=positive, required=True)
    commands.add_parser('list')
    command = commands.add_parser('show')
    command.add_argument('id')
    for name in ('accept', 'reject', 'defer', 'edit'):
        command = commands.add_parser(name)
        command.add_argument('id')
        command.add_argument('--expected-sha', required=True)
        command.add_argument('--reason', default='' if name == 'defer' else None, required=name != 'defer')
        command.add_argument('--acknowledge-flags', action='store_true')
        if name == 'accept':
            command.add_argument('--answer', choices=('left', 'right'), default='left',
                                 help='Preferred answer: left is proposed, right is historical')
        if name == 'edit':
            command.add_argument('--body-file', required=True)
    command = commands.add_parser('render')
    command.add_argument('--preview', action='store_true')
    short = commands.add_parser('short', help='Draft and review short session-start examples')
    short_commands = short.add_subparsers(dest='short_cmd', required=True)
    short_commands.add_parser('list')
    command = short_commands.add_parser('show')
    command.add_argument('id')
    command = short_commands.add_parser('generate', help='Retry model drafting for an accepted pair')
    command.add_argument('pair_id')
    command.add_argument('--expected-pair-sha', required=True)
    command = short_commands.add_parser('draft')
    command.add_argument('pair_id')
    command.add_argument('--expected-pair-sha', required=True)
    command.add_argument('--expected-sha')
    command.add_argument('--body-file', required=True, help='JSON with situation, preferred, avoid, feedback, limits')
    for name in ('approve', 'reject'):
        command = short_commands.add_parser(name)
        command.add_argument('id')
        command.add_argument('--expected-sha', required=True)
        command.add_argument('--reason', required=True)
    command = commands.add_parser('export')
    command.add_argument('--format', choices=('dpo-jsonl',), required=True)
    command.add_argument('--preview', action='store_true')
    command = commands.add_parser('eval')
    group = command.add_mutually_exclusive_group(required=True)
    group.add_argument('--fixtures')
    group.add_argument('--report')
    command.add_argument('--limit', type=positive)
    parser.set_defaults(func=run)


def _save(config, result, outputs, manifest_name):
    manifest = dict(result['manifest'])
    manifest['files'] = {name: hashlib.sha256(body.encode('utf-8')).hexdigest() for name, body in outputs.items()}
    write_artifacts(config, {**outputs, manifest_name: canonical_json(manifest) + '\n'}, manifest)
    return {'files': list(outputs), 'manifest': manifest_name}


def _initialize_for_mutation(config):
    """Perform the approved host schema migration before selecting model work."""
    from character.constitution import load_constitution
    from character.storage import open_writer
    ready = False
    if config.db_path.exists():
        reader = open_db_readonly(config.db_path)
        try:
            migrations = reader.execute("SELECT 1 FROM sqlite_master WHERE name='schema_migrations'").fetchone()
            ready = bool(migrations and reader.execute('SELECT COALESCE(MAX(version),0) FROM schema_migrations').fetchone()[0] >= 4)
        finally:
            reader.close()
    if not ready:
        load_constitution(config)
        with open_writer(config):
            pass


def run(args):
    cfg = args.config
    cfg = replace(cfg, data={**cfg.data, 'storage': {**cfg.data['storage'], 'db_path': str(args.db)}})
    action = args.character_cmd
    if (action in ('generate', 'judge') or action == 'short' and args.short_cmd == 'generate'
            or action == 'eval' and not args.report) and not cfg.data['character']['enabled']:
        print(json.dumps({'skipped': 'character disabled'}))
        return 0
    conn = None
    try:
        guard_path(cfg.db_path, cfg)
        mutation = (action in ('generate', 'judge', 'accept', 'reject', 'defer', 'edit')
                    or action == 'short' and args.short_cmd in ('draft', 'generate', 'approve', 'reject')
                    or action == 'eval' and not args.report)
        if mutation:
            _initialize_for_mutation(cfg)
        conn = open_db_readonly(cfg.db_path)
        if action in ('estimate', 'generate'):
            from character.generate import estimate, generate_pairs
            method = estimate if action == 'estimate' else generate_pairs
            result = method(conn, cfg, source=args.source, limit=args.limit,
                            refresh=args.refresh, expand_seeds=args.expand_seeds)
        elif action == 'judge':
            from character.judge import judge_pairs
            result = judge_pairs(conn, cfg, limit=args.limit)
        elif action in ('list', 'show', 'accept', 'reject', 'defer', 'edit'):
            from character.review import list_pairs, show_pair, review_pair
            if action == 'list':
                result = list_pairs(conn, cfg)
            elif action == 'show':
                result = show_pair(conn, cfg, args.id)
            else:
                body = Path(args.body_file).read_text(encoding='utf-8') if action == 'edit' else None
                selection = {'answer': args.answer} if action == 'accept' else {}
                result = review_pair(conn, cfg, args.id, action, expected_sha=args.expected_sha,
                                     reason=args.reason, body=body, acknowledge_flags=args.acknowledge_flags,
                                     **selection)
                if action == 'accept':
                    from character.short import generate_example, draft_failure
                    conn.close()
                    conn = None
                    try:
                        conn = open_db_readonly(cfg.db_path)
                        draft = generate_example(conn, cfg, args.id, expected_pair_sha=result['content_sha256'])
                    except (OSError, sqlite3.Error) as exc:
                        draft = draft_failure(args.id, result['content_sha256'], exc)
                    result = dict(result, short_draft=draft)
                    if draft.get('failures'):
                        result['failures'] = draft['failures']
        elif action == 'short':
            from character.short import list_examples, show_example, draft_example, review_example, generate_example
            if args.short_cmd == 'list':
                result = list_examples(conn, cfg)
            elif args.short_cmd == 'show':
                result = show_example(conn, cfg, args.id)
            elif args.short_cmd == 'generate':
                result = generate_example(conn, cfg, args.pair_id, expected_pair_sha=args.expected_pair_sha)
            elif args.short_cmd == 'draft':
                body = json.loads(Path(args.body_file).read_text(encoding='utf-8'))
                result = draft_example(conn, cfg, args.pair_id, expected_pair_sha=args.expected_pair_sha,
                                       expected_sha=args.expected_sha, body=body)
            else:
                result = review_example(conn, cfg, args.id, args.short_cmd,
                                        expected_sha=args.expected_sha, reason=args.reason)
        elif action in ('render', 'export'):
            if action == 'render':
                from character.render import render_block
                result = render_block(conn, cfg)
                filename = 'block.md'
            else:
                from character.export import export_pairs
                result = export_pairs(conn, cfg)
                filename = 'accepted.dpo.jsonl'
            if not args.preview:
                result = _save(cfg, result, {filename: result['text']}, filename + '.manifest.json')
        else:
            from character.eval import evaluate, report_run
            if args.report:
                result = report_run(conn, cfg, args.report)
            else:
                if args.limit is None:
                    raise ValueError('eval requires --limit before model calls')
                result = evaluate(conn, cfg, fixtures=Path(args.fixtures), limit=args.limit)
                if result.get('run_id'):
                    conn.close()
                    conn = open_db_readonly(cfg.db_path)
                    report = report_run(conn, cfg, result['run_id'])
                    stem = 'eval-' + result['run_id']
                    _save(cfg, report, {stem + '.json': canonical_json(report), stem + '.md': report['markdown']}, stem + '.manifest.json')
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1 if isinstance(result, dict) and result.get('failures') else 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f'character {action}: {type(exc).__name__}: {str(exc)[:500]}', file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()
