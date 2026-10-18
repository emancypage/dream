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
        if name == 'edit':
            command.add_argument('--body-file', required=True)
    command = commands.add_parser('render')
    command.add_argument('--preview', action='store_true')
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


def run(args):
    cfg = args.config
    cfg = replace(cfg, data={**cfg.data, 'storage': {**cfg.data['storage'], 'db_path': str(args.db)}})
    action = args.character_cmd
    if action in ('generate', 'judge') and not cfg.data['character']['enabled']:
        print(json.dumps({'skipped': 'character disabled'}))
        return 0
    conn = None
    try:
        guard_path(cfg.db_path, cfg)
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
                result = review_pair(conn, cfg, args.id, action, expected_sha=args.expected_sha,
                                     reason=args.reason, body=body, acknowledge_flags=args.acknowledge_flags)
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
