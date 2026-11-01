#!/usr/bin/env python3
"""Local stdio MCP App. stdout is reserved for JSON-RPC messages."""
import base64
import contextlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def load_runtime():
    executable = shutil.which('dream')
    if executable is None:
        raise RuntimeError('Install Dream first; its dream executable must be on PATH.')
    runtime = Path(executable).resolve().parent
    if not (runtime / 'config.py').is_file() or not (runtime / 'character' / 'review.py').is_file():
        raise RuntimeError('The installed dream executable does not point to the Dream Python source.')
    sys.path.insert(0, str(runtime))


def schema(properties=None, required=None):
    return dict(type='object', properties=properties or {}, required=required or [], additionalProperties=False)


def validate(value, spec, path='arguments'):
    kind = spec.get('type')
    valid = {'object': isinstance(value, dict), 'string': isinstance(value, str),
             'boolean': type(value) is bool}
    if kind and not valid.get(kind, False):
        raise ValueError(f'{path}: expected {kind}')
    if 'enum' in spec and value not in spec['enum']:
        raise ValueError(f'{path}: invalid choice')
    if kind == 'string':
        if not spec.get('minLength', 0) <= len(value) <= spec.get('maxLength', 100000):
            raise ValueError(f'{path}: invalid length')
    if kind == 'object':
        props = spec['properties']
        if set(value) - set(props) or set(spec.get('required', [])) - set(value):
            raise ValueError(f'{path}: missing or unknown fields')
        for key, item in value.items():
            validate(item, props[key], path + '.' + key)


class ReviewServer:
    uri = 'ui://dream-review/panel.html'

    def __init__(self, app):
        self.app = app
        self.views = {}

    def tools(self):
        text = dict(type='string', minLength=1, maxLength=200)
        sha = dict(type='string', minLength=64, maxLength=64)
        kind = dict(type='string', enum=['pair', 'short'])
        reason = dict(type='string', minLength=1, maxLength=2000)
        identity = dict(kind=kind, id=text)
        decision = dict(**identity, expected_sha=sha, view_token=text,
                        choice=dict(type='string', enum=['left', 'right', 'reject', 'defer', 'approve']),
                        reason=reason, acknowledge_flags=dict(type='boolean'))
        body = schema({key: dict(type='string', minLength=1, maxLength=2500)
                       for key in ('situation', 'preferred', 'avoid', 'feedback', 'limits')},
                      ['situation', 'preferred', 'avoid', 'feedback', 'limits'])
        edit = dict(id=text, expected_sha=sha, expected_pair_sha=sha, view_token=text, body=body)
        prepare = dict(id=text, expected_sha=sha, view_token=text)
        deletion = dict(**prepare, reason=dict(type='string', maxLength=2000))
        specs = [
            ('show_review', 'Dream Review', 'Open the local Dream human review panel beside this chat.', schema(), True),
            ('get_queue', 'Read review queue', 'Read current pairs, short drafts and review history.', schema(), True),
            ('get_item', 'Read review item', 'Display complete evidence and bind a decision to its exact version.', schema(identity, ['kind', 'id']), True),
            ('save_decision', 'Save human decision', 'Record the human choice for the exact displayed version. Never decide for the human.', schema(decision, list(decision)), False),
            ('edit_short', 'Save short draft', 'Save revised wording as an unapproved short draft. Approval remains separate.', schema(edit, list(edit)), False),
            ('prepare_short', 'Prepare short draft', 'Retry preparing an unapproved short draft for an accepted pair.', schema(prepare, list(prepare)), False),
            ('delete_pair', 'Delete pair permanently', 'Remove the displayed pair and block regeneration. Optional human reason informs future memory curation.', schema(deletion, list(prepare)), False),
        ]
        tools = []
        for name, title, description, input_schema, readonly in specs:
            meta = {'ui': {'visibility': ['model', 'app'] if name == 'show_review' else ['app']}}
            if name == 'show_review':
                meta['ui']['resourceUri'] = self.uri
                meta['openai/ui'] = {'entrypoints': [{'type': 'global'}, {'type': 'thread'}]}
            tool = dict(name=name, title=title, description=description, inputSchema=input_schema,
                annotations=dict(readOnlyHint=readonly, destructiveHint=name == 'delete_pair', openWorldHint=False,
                                 idempotentHint=readonly), _meta=meta)
            if name == 'show_review':
                icon = base64.b64encode((ROOT / 'assets' / 'icon.svg').read_bytes()).decode('ascii')
                tool['icons'] = [dict(src='data:image/svg+xml;base64,' + icon,
                                      mimeType='image/svg+xml', sizes=['any'])]
            tools.append(tool)
        return tools

    def _view(self, kind, identity, sha, token):
        expected = self.views.get((kind, identity, sha))
        if expected is None or not hmac.compare_digest(expected, token):
            raise ValueError('Read and display this exact version before saving a decision.')

    def call(self, name, args):
        tool = next((t for t in self.tools() if t['name'] == name), None)
        if tool is None:
            raise ValueError('Unknown Dream Review tool')
        validate(args, tool['inputSchema'])
        if name in ('show_review', 'get_queue'):
            result = self.app.queue()
            return {'counts': result['counts']} if name == 'show_review' else result
        if name == 'get_item':
            result = self.app.detail(**args)
            token = secrets.token_urlsafe(32)
            if len(self.views) >= 100:
                self.views.pop(next(iter(self.views)))
            self.views[(args['kind'], args['id'], result['content_sha256'])] = token
            return dict(result, view_token=token)
        kind = 'pair' if name in ('prepare_short', 'delete_pair') else args.get('kind', 'short')
        self._view(kind, args['id'], args['expected_sha'], args['view_token'])
        values = {key: value for key, value in args.items() if key != 'view_token'}
        if name == 'save_decision':
            result = self.app.decide(**values)
        elif name == 'prepare_short':
            result = self.app.prepare_short(**values)
        elif name == 'delete_pair':
            result = self.app.delete_pair(**values)
        else:
            result = self.app.edit_short(**values)
        self.views.pop((kind, args['id'], args['expected_sha']), None)
        return result

    def resource(self):
        return dict(uri=self.uri, name='dream-review', title='Dream Review', mimeType='text/html;profile=mcp-app',
                    _meta={'ui': {'prefersBorder': False,
                                 'csp': {'connectDomains': [], 'resourceDomains': []},
                                 'preferredDisplayMode': 'fullscreen',
                                 'availableDisplayModes': ['inline', 'fullscreen']}})

    def handle(self, message):
        if not isinstance(message, dict) or message.get('jsonrpc') != '2.0' or not isinstance(message.get('method'), str):
            return dict(jsonrpc='2.0', id=None, error=dict(code=-32600, message='Invalid request'))
        if 'id' not in message:
            return None
        identity = message['id']
        method = message['method']
        params = message.get('params', {})
        if not isinstance(params, dict):
            return dict(jsonrpc='2.0', id=identity, error=dict(code=-32602, message='Invalid parameters'))
        try:
            if method == 'initialize':
                version = params.get('protocolVersion', '2025-11-25')
                if version not in ('2026-07-28', '2025-11-25', '2025-06-18'):
                    version = '2025-11-25'
                result = dict(protocolVersion=version, capabilities=dict(tools={}, resources={}),
                    serverInfo=dict(name='dream-review', version='0.1.6'),
                    instructions='Open show_review when asked. Human decisions are made in the panel. Never choose or approve character examples autonomously.')
            elif method == 'ping':
                result = {}
            elif method == 'tools/list':
                result = dict(tools=self.tools())
            elif method == 'tools/call':
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        value = self.call(params.get('name'), params.get('arguments', {}))
                    result = dict(content=[dict(type='text', text=json.dumps(value, ensure_ascii=False))], structuredContent=value)
                except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
                    result = dict(isError=True, content=[dict(type='text', text=str(exc)[:1000])])
            elif method == 'resources/list':
                result = dict(resources=[self.resource()])
            elif method == 'resources/templates/list':
                result = dict(resourceTemplates=[])
            elif method == 'resources/read':
                if params.get('uri') != self.uri:
                    raise ValueError('Unknown UI resource')
                result = dict(contents=[dict(**self.resource(), text=(ROOT / 'panel.html').read_text(encoding='utf-8'))])
            else:
                return dict(jsonrpc='2.0', id=identity, error=dict(code=-32601, message='Method not found'))
            return dict(jsonrpc='2.0', id=identity, result=result)
        except (ValueError, KeyError, TypeError):
            return dict(jsonrpc='2.0', id=identity, error=dict(code=-32602, message='Invalid parameters'))
        except Exception:
            print('Dream Review request failed', file=sys.stderr)
            return dict(jsonrpc='2.0', id=identity, error=dict(code=-32603, message='Dream Review request failed'))


def main():
    load_runtime()
    from review_app import ReviewApp
    instance = ReviewServer(ReviewApp())
    while True:
        line = sys.stdin.buffer.readline(1024 * 1024 + 1)
        if not line:
            return 0
        if len(line) > 1024 * 1024:
            while line and not line.endswith(b'\n'):
                line = sys.stdin.buffer.readline(65536)
            result = dict(jsonrpc='2.0', id=None, error=dict(code=-32700, message='Message too large'))
        else:
            try:
                result = instance.handle(json.loads(line))
            except (ValueError, UnicodeError):
                result = dict(jsonrpc='2.0', id=None, error=dict(code=-32700, message='Invalid JSON'))
        if result is not None:
            try:
                sys.stdout.write(json.dumps(result, ensure_ascii=True, allow_nan=False) + '\n')
                sys.stdout.flush()
            except BrokenPipeError:
                return 0


if __name__ == '__main__':
    raise SystemExit(main())
