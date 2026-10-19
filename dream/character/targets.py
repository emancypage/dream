"""Provider target contract for controlled character evaluations."""
from typing import Protocol
import backend
from character.types import RESPONSE_SCHEMA, canonical_json

class Target(Protocol):
    def identity(self) -> dict: ...
    def respond(self, messages, context_block, request_settings): ...

class ProviderTarget:
    """Baseline and A use one route; context is the only treatment."""
    def __init__(self, config):
        self.config = config

    def identity(self):
        return dict(self.config.stage('character_eval_response'))

    def respond(self, messages, context_block, request_settings):
        system = request_settings['instructions']
        if context_block:
            system += '\n' + context_block
        system += request_settings['policy_context']
        return backend.generate('character_eval_response', canonical_json({'messages': messages}),
                                RESPONSE_SCHEMA, config=self.config, system_prompt=system)


def make_target(config):
    """Adapter factory; future targets can replace the transport here."""
    return ProviderTarget(config)
