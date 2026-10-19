"""Strict parsing of the human-owned constitution."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import re
from types import MappingProxyType

@dataclass(frozen=True)
class Constitution:
    text: str
    sha256: str
    version: str
    traits: object
    seeds: tuple
    approved_at: str | None
    status: str


def load_constitution(config, require_approved=True):
    path = Path(config.data['character']['constitution_path']).expanduser()
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    lines = text.splitlines()
    if not lines or lines[0] != '---':
        raise ValueError('constitution requires frontmatter')
    try:
        end = lines.index('---', 1)
    except ValueError:
        raise ValueError('unterminated constitution frontmatter') from None
    fields = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(':')
        if not sep or key not in {'status', 'version', 'approved_at'} or key in fields:
            raise ValueError('invalid constitution frontmatter field')
        value = value.strip()
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError('invalid constitution scalar')
            value = value[1:-1]
        fields[key] = value
    if fields.get('status') not in {'draft', 'approved'} or not fields.get('version'):
        raise ValueError('invalid constitution status or version')
    approved = fields.get('approved_at') or None
    if fields['status'] == 'approved':
        try:
            stamp = datetime.fromisoformat((approved or '').replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                raise ValueError()
        except ValueError:
            raise ValueError('approved constitution requires ISO approval timestamp') from None
    if require_approved and fields['status'] != 'approved':
        raise ValueError('constitution must be approved by a human')
    sections = {}
    current = None
    for line in lines[end + 1:]:
        if line.startswith('## '):
            current = line[3:].strip()
            if current in sections:
                raise ValueError('duplicate constitution section')
            sections[current] = []
        elif current:
            sections[current].append(line)
    required = ('Traits', 'Priority order', 'Hard constraints', 'Seed prompts per trait')
    if any(name not in sections for name in required[:3]):
        raise ValueError('missing constitution section')
    traits = {}
    blocks = re.split(r'^### (trait-[A-Za-z0-9_-]+)\s*$', '\n'.join(sections['Traits']), flags=re.M)
    for i in range(1, len(blocks), 2):
        key, body = blocks[i], blocks[i + 1].strip()
        if key in traits:
            raise ValueError('duplicate trait ID')
        reason = re.search(r'^Why:\s*(.+)$', body, re.M)
        if fields['status'] == 'approved' and (not reason or not body.split('Why:')[0].strip() or 'TODO' in body):
            raise ValueError('trait requires text and Why reason')
        traits[key] = body
    if fields['status'] == 'approved':
        if not traits:
            raise ValueError('constitution requires traits')
        for name in ('Priority order', 'Hard constraints'):
            body = '\n'.join(sections[name]).strip()
            if not body or 'TODO' in body:
                raise ValueError('unresolved required constitution section')
    seeds = []
    blocks = re.split(r'^### (seed-[A-Za-z0-9_-]+)\s*$', '\n'.join(sections.get(required[-1], [])), flags=re.M)
    for i in range(1, len(blocks), 2):
        key, body = blocks[i], blocks[i + 1].strip()
        trait = re.search(r'^Trait:\s*(trait-[A-Za-z0-9_-]+)\s*$', body, re.M)
        if not body or body == 'TODO':
            continue
        if not trait or trait[1] not in traits or any(s['id'] == key for s in seeds):
            raise ValueError('invalid seed ID or trait reference')
        prompt = body[trait.end():].strip()
        if prompt and prompt != 'TODO':
            seeds.append(MappingProxyType({'id': key, 'trait_id': trait[1], 'prompt': prompt}))
    return Constitution(text, hashlib.sha256(raw).hexdigest(), fields['version'], MappingProxyType(traits), tuple(seeds), approved, fields['status'])
