"""Canonical host record helpers and shared provider schemas."""
import hashlib
import json

REVIEW_STATUSES = ('pending', 'deferred', 'edited', 'accepted', 'rejected', 'stale')
RESPONSE_SCHEMA = {'type': 'object', 'properties': {'response': {'type': 'string'}}, 'required': ['response'], 'additionalProperties': False}

def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()
