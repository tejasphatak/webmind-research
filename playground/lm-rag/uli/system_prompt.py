"""
ULI System Configuration + Safety Gate.

Loads data/system/config.yaml once at startup (singleton).
SafetyGate.check() runs on every input before routing.

Usage:
    from uli.system_prompt import SystemConfig, SafetyGate

    cfg = SystemConfig.load()
    gate = SafetyGate(cfg.settings)
    is_safe, reason = gate.check(user_text)
    if not is_safe:
        return reason  # return to user, don't answer
"""

import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger('uli.system_prompt')

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'system', 'config.yaml'
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MCPServerConfig:
    name: str
    transport: str                        # 'sse' | 'stdio' | 'builtin' | 'rest_adapter'
    capabilities: List[str]
    url: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    timeout_seconds: int = 8
    api_key_env: Optional[str] = None
    enabled: bool = True


@dataclass
class TrustTier:
    name: str
    score: float
    sources: List[str]
    auto_store: bool


@dataclass
class SystemSettings:
    name: str
    version: str
    system_prompt: str
    capabilities: List[str]
    max_input_chars: int
    blocklist_patterns: List[re.Pattern]   # pre-compiled
    crisis_patterns: List[re.Pattern]      # pre-compiled
    crisis_response: str
    mcp_servers: List[MCPServerConfig]
    trust_tiers: List[TrustTier]
    corroboration_boost: float
    store_threshold: float


# ── Singleton loader ──────────────────────────────────────────────────────────

class SystemConfig:
    _instance: Optional['SystemConfig'] = None
    _settings: Optional[SystemSettings] = None

    def __init__(self, settings: SystemSettings):
        self.settings = settings

    @classmethod
    def load(cls, path: str = None) -> 'SystemConfig':
        """Load config from YAML. Singleton — returns cached instance on repeat calls."""
        if cls._instance is not None:
            return cls._instance

        cfg_path = path or os.environ.get('ULI_CONFIG', _DEFAULT_CONFIG_PATH)
        raw = cls._read_yaml(cfg_path)
        settings = cls._parse(raw)
        cls._instance = cls(settings)
        log.info(f"SystemConfig loaded from {cfg_path} — "
                 f"{len(settings.mcp_servers)} MCP servers, "
                 f"{len(settings.blocklist_patterns)} blocklist patterns")
        return cls._instance

    @classmethod
    def reset(cls):
        """Clear singleton — for testing only."""
        cls._instance = None

    @classmethod
    def _read_yaml(cls, path: str) -> dict:
        """Load YAML file. Falls back to empty dict if file or yaml missing."""
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            log.warning("pyyaml not installed — using hardcoded defaults")
            return {}
        except FileNotFoundError:
            log.warning(f"Config not found at {path} — using hardcoded defaults")
            return {}

    @classmethod
    def _parse(cls, raw: dict) -> SystemSettings:
        identity   = raw.get('identity', {})
        safety_raw = raw.get('safety', {})
        trust_raw  = raw.get('source_trust', {})

        # Pre-compile regex patterns (do it once here, not at check() time)
        blocklist = [
            re.compile(p, re.IGNORECASE)
            for p in safety_raw.get('blocklist_patterns', [])
        ]
        crisis = [
            re.compile(p, re.IGNORECASE)
            for p in safety_raw.get('crisis_patterns', [])
        ]

        # Parse MCP server configs
        mcp_servers = []
        for srv in raw.get('mcp_servers', []):
            mcp_servers.append(MCPServerConfig(
                name=srv['name'],
                transport=srv['transport'],
                capabilities=srv.get('capabilities', []),
                url=srv.get('url'),
                tools=srv.get('tools', []),
                timeout_seconds=srv.get('timeout_seconds', 8),
                api_key_env=srv.get('api_key_env'),
                enabled=srv.get('enabled', True),
            ))

        # Parse trust tiers
        trust_tiers = []
        for tier in trust_raw.get('tiers', []):
            trust_tiers.append(TrustTier(
                name=tier['name'],
                score=tier['score'],
                sources=tier.get('sources', []),
                auto_store=tier.get('auto_store', False),
            ))

        # Hardcoded defaults if YAML missing or incomplete
        return SystemSettings(
            name=identity.get('name', 'ULI Reasoning Engine'),
            version=identity.get('version', '1.0'),
            system_prompt=identity.get('system_prompt', (
                "You are a knowledge reasoning assistant. "
                "Answer from facts, not speculation."
            )),
            capabilities=raw.get('capabilities', ['factual_qa', 'web_research']),
            max_input_chars=safety_raw.get('max_input_chars', 8000),
            blocklist_patterns=blocklist,
            crisis_patterns=crisis,
            crisis_response=safety_raw.get('crisis_response', (
                "Please reach out to a crisis helpline. "
                "US: 988 Suicide & Crisis Lifeline (call or text 988)."
            )),
            mcp_servers=mcp_servers,
            trust_tiers=trust_tiers,
            corroboration_boost=trust_raw.get('corroboration_boost', 0.15),
            store_threshold=trust_raw.get('store_threshold', 0.75),
        )

    def trust_score(self, source_name: str) -> float:
        """Return trust score for a given source name."""
        for tier in self.settings.trust_tiers:
            if source_name in tier.sources:
                return tier.score
        return 0.30  # unknown

    def auto_store(self, source_name: str) -> bool:
        """Return True if results from this source should be auto-stored in KG."""
        for tier in self.settings.trust_tiers:
            if source_name in tier.sources:
                return tier.auto_store
        return False

    def mcp_servers_for(self, capability: str) -> List[MCPServerConfig]:
        """Return enabled servers that have the given capability, in config order."""
        return [
            s for s in self.settings.mcp_servers
            if s.enabled and capability in s.capabilities
        ]


# ── Safety Gate ───────────────────────────────────────────────────────────────

class SafetyGate:
    """
    Pre-routing content filter. Called before any processing.

    check(text) returns (is_safe: bool, message: str).
    is_safe=False → return message to user immediately, skip all reasoning.
    """

    def __init__(self, settings: SystemSettings):
        self._settings = settings

    def check(self, text: str) -> Tuple[bool, str]:
        """
        Check input text for safety violations.

        Returns:
            (True, '')          — safe, proceed
            (False, message)    — unsafe, return message to user
        """
        if not text or not text.strip():
            return True, ''

        # 1. Length guard
        if len(text) > self._settings.max_input_chars:
            return False, (
                f"Input too long ({len(text)} chars). "
                f"Please keep questions under {self._settings.max_input_chars} characters."
            )

        # 2. Crisis detection (highest priority — respond with support, not refusal)
        for pattern in self._settings.crisis_patterns:
            if pattern.search(text):
                return False, self._settings.crisis_response

        # 3. Blocklist patterns
        for pattern in self._settings.blocklist_patterns:
            if pattern.search(text):
                return False, (
                    "I'm not able to help with that. If you have another question, "
                    "I'm here to help."
                )

        return True, ''
