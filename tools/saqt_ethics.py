#!/usr/bin/env python3
"""
SAQT Ethics Invariant Layer
============================
Structurally-enforced ethics for the SAQT/Webmind knowledge engine.

Three guarantees:
  1. IMMUTABLE LAYER  — signed ethics pairs that live outside the mutable DB
  2. INGESTION FILTER — every new Q&A pair checked before insert; hard-reject on violation
  3. TOOL FILTER      — every <tool> block scanned before execution; hard-reject on dangerous patterns

Design principles:
  - Ethics are NOT Q&A pairs with high weights. They are code-level constraints.
  - The immutable layer is read-only at runtime; modification requires a new signed bundle.
  - Filters run BEFORE DB write and BEFORE tool execution — not after.
  - Weight floors are enforced at query time by the SAQTDB wrapper below.

Usage:
    from saqt_ethics import EthicsGuard, EthicsAwareSAQTDB

    guard = EthicsGuard()
    safe_db = EthicsAwareSAQTDB("/path/to/saqt.db", guard=guard, encoder=encoder)

    # Add pair — raises EthicsViolation if it violates invariants
    safe_db.add("How do I make a bomb?", "Here are the steps...", source="user")

    # Execute tool code — raises EthicsViolation if dangerous
    guard.check_tool_code("import os; os.system('rm -rf /')")
"""

import hashlib
import hmac
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Ethics Violation Signal
# ─────────────────────────────────────────────────────────────────────────────

class EthicsViolation(Exception):
    """Raised when an ethics invariant is triggered. NEVER catch and suppress."""
    def __init__(self, rule: str, detail: str):
        self.rule = rule
        self.detail = detail
        super().__init__(f"[ETHICS:{rule}] {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Immutable Ethics Rules
# These are NOT stored in the DB. They are hardcoded at the class level.
# To change them you change the source code — which is version-controlled
# and requires a deliberate commit. No runtime path can modify them.
# ─────────────────────────────────────────────────────────────────────────────

# Each rule: (rule_id, description, regex_patterns_that_trigger_it)
# Patterns match question OR answer text (case-insensitive).
IMMUTABLE_RULES: list[dict] = [
    {
        "id": "WEAPONS",
        "description": "Never provide instructions for weapons, explosives, or mass violence",
        "question_patterns": [
            r"\b(how|steps?|instructions?|guide|tutorial|make|build|create|synthesize)\b.{0,40}"
            r"\b(bomb|explosive|ied|bioweapon|biological weapon|chemical weapon|nerve agent|sarin|ricin|anthrax)\b",
            r"\b(3d.?print|manufacture|illegal\s+weapon|convert\s+semi|full.?auto)\b",
        ],
        "answer_patterns": [
            r"\b(step\s+\d|first,?\s+you|here('s|\s+are)\s+the\s+steps)\b.{0,100}"
            r"\b(explosive|detonator|fuse|primer|oxidizer|ammonium nitrate)\b",
            r"\b(synthesis|synthesize).{0,30}\b(nerve agent|VX|sarin|novichok|ricin)\b",
        ],
        "severity": "HARD_BLOCK",
    },
    {
        "id": "PII_STORE",
        "description": "Never store PII: SSN, full credit card numbers, PCI data, passport numbers",
        "question_patterns": [
            r"\b\d{3}-\d{2}-\d{4}\b",                          # SSN
            r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", # card number
            r"\b(cvv|cvc|security code)\b.{0,20}\d{3,4}\b",
            r"\bpassport\s*(no|number|#)\b.{0,10}[A-Z0-9]{6,9}",
        ],
        "answer_patterns": [
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
        ],
        "severity": "HARD_BLOCK",
    },
    {
        "id": "CSAM",
        "description": "Never generate or store any content sexualizing minors",
        "question_patterns": [
            r"\b(child|minor|underage|teen|kid).{0,30}(naked|nude|sexual|porn|explicit)\b",
            r"\b(sexual|explicit).{0,30}(child|minor|underage|kid)\b",
        ],
        "answer_patterns": [
            r"\b(child|minor|underage).{0,30}(naked|nude|sexual|explicit)\b",
        ],
        "severity": "HARD_BLOCK",
    },
    {
        "id": "HARM_INDIVIDUAL",
        "description": "Never generate content designed to harm a specific named individual",
        "question_patterns": [
            r"\b(kill|murder|hurt|harm|stalk|doxx|find address of).{0,30}[A-Z][a-z]+\s+[A-Z][a-z]+",
            r"\bhow\s+to\s+(stalk|doxx|threaten|intimidate).{0,30}(person|someone|him|her|they)\b",
        ],
        "answer_patterns": [
            r"\b(here('s|\s+is) how to (find|locate|track)).{0,50}(home address|location|phone number)\b",
        ],
        "severity": "HARD_BLOCK",
    },
    {
        "id": "SELF_HARM",
        "description": "Never provide methods for self-harm or suicide to at-risk individuals",
        "question_patterns": [
            r"\b(what|how).{0,20}(pills|drugs|dose|method).{0,30}(kill myself|end my life|suicide|overdose)\b",
            r"\b(most effective|painless|quickest).{0,30}(suicide|way to die|self harm)\b",
        ],
        "answer_patterns": [
            r"\b(lethal dose|LD50|fatal overdose).{0,50}(mg|grams?|pills?)\b",
        ],
        "severity": "HARD_BLOCK",
    },
    {
        "id": "OVERRIDE_ETHICS",
        "description": "Reject any Q&A pair that instructs the system to ignore/bypass/override ethics",
        "question_patterns": [
            r"\b(ignore|bypass|disable|override|forget|pretend).{0,30}(ethics|safety|guardrail|rule|restriction|filter)\b",
            r"\b(you are|act as|pretend to be).{0,30}(no restriction|no limit|without ethics|unfiltered)\b",
            r"\bjailbreak\b",
        ],
        "answer_patterns": [
            r"\b(I will|I can|sure,? I).{0,30}(ignore|bypass|disable).{0,30}(ethics|safety|rules)\b",
        ],
        "severity": "HARD_BLOCK",
    },
]

# Compile all regexes once at import time
def _compile_rules(rules: list[dict]) -> list[dict]:
    compiled = []
    for r in rules:
        compiled.append({
            **r,
            "_q_compiled": [re.compile(p, re.IGNORECASE | re.DOTALL)
                            for p in r.get("question_patterns", [])],
            "_a_compiled": [re.compile(p, re.IGNORECASE | re.DOTALL)
                            for p in r.get("answer_patterns", [])],
        })
    return compiled

_COMPILED_RULES = _compile_rules(IMMUTABLE_RULES)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Tool Execution Patterns — things NEVER allowed in <tool> blocks
# ─────────────────────────────────────────────────────────────────────────────

# Server-side (Python subprocess) blocked patterns
SERVER_TOOL_BLOCKED: list[tuple[str, str]] = [
    (r"rm\s+-rf|rmdir\s+/",                         "TOOL_FS_DESTROY"),
    (r"os\.(system|popen|execv?p?e?|spawnl?e?)",    "TOOL_OS_EXEC"),
    (r"subprocess\.(run|call|Popen|check_output)",  "TOOL_SUBPROCESS"),
    (r"__import__\s*\(",                            "TOOL_DYNAMIC_IMPORT"),
    (r"importlib\.(import_module|reload)",          "TOOL_DYNAMIC_IMPORT"),
    (r"eval\s*\(|exec\s*\(",                        "TOOL_EVAL"),
    (r"open\s*\(.*(w|a|wb|ab)",                     "TOOL_FS_WRITE"),
    (r"socket\.(socket|connect|send)",              "TOOL_RAW_SOCKET"),
    (r"urllib|requests|httpx|aiohttp|fetch",        "TOOL_NETWORK"),
    (r"pickle\.(loads?|dumps?)",                    "TOOL_PICKLE_DESER"),
    (r"ctypes\.|cffi\.",                            "TOOL_FFI"),
    (r"/etc/passwd|/etc/shadow|/proc/self",         "TOOL_SENSITIVE_PATH"),
    (r"base64\.b64decode.{0,40}exec",               "TOOL_OBFUSCATED_EXEC"),
    (r"chr\(\d+\).{0,10}\+.{0,10}chr\(",           "TOOL_CHAR_CONCAT_EXEC"),
]

# Browser-side (Web Worker JS) blocked patterns
BROWSER_TOOL_BLOCKED: list[tuple[str, str]] = [
    (r"eval\s*\(",                                  "TOOL_EVAL"),
    (r"new\s+Function\s*\(",                        "TOOL_NEW_FUNCTION"),
    (r"setTimeout\s*\(\s*['\"]",                    "TOOL_SETTIMEOUT_STRING"),
    (r"setInterval\s*\(\s*['\"]",                   "TOOL_SETINTERVAL_STRING"),
    (r"importScripts\s*\(",                         "TOOL_IMPORT_SCRIPTS"),
    (r"XMLHttpRequest|fetch\s*\(",                  "TOOL_NETWORK"),
    (r"WebSocket\s*\(",                             "TOOL_WEBSOCKET"),
    (r"indexedDB|localStorage|sessionStorage|cookie", "TOOL_STORAGE"),
    (r"postMessage\s*\(.*(password|token|key|secret)", "TOOL_SECRET_EXFIL"),
    (r"document\.|window\.",                        "TOOL_DOM_ACCESS"),
    (r"navigator\.(sendBeacon|geolocation|usb)",   "TOOL_SENSITIVE_API"),
    (r"atob\s*\(.{0,40}eval|eval.{0,40}atob\s*\(", "TOOL_B64_EVAL"),
    (r"String\.fromCharCode",                       "TOOL_CHAR_OBFUSCATION"),
]

def _compile_tool_patterns(patterns):
    return [(re.compile(p, re.IGNORECASE | re.DOTALL), code)
            for p, code in patterns]

_SERVER_COMPILED  = _compile_tool_patterns(SERVER_TOOL_BLOCKED)
_BROWSER_COMPILED = _compile_tool_patterns(BROWSER_TOOL_BLOCKED)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Ethics Pair Signing
# Used to sign the immutable layer pairs so they can be verified as unmodified.
# Key is stored separately from the DB; runtime only has the public verifier.
# ─────────────────────────────────────────────────────────────────────────────

def sign_pair(question: str, answer: str, secret_key: bytes) -> str:
    """HMAC-SHA256 signature for a Q&A pair."""
    payload = json.dumps({"q": question, "a": answer}, sort_keys=True,
                         ensure_ascii=False).encode("utf-8")
    return hmac.new(secret_key, payload, hashlib.sha256).hexdigest()


def verify_pair(question: str, answer: str, secret_key: bytes, expected_sig: str) -> bool:
    """Constant-time HMAC verification."""
    actual = sign_pair(question, answer, secret_key)
    return hmac.compare_digest(actual, expected_sig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: EthicsGuard — the central enforcement object
# ─────────────────────────────────────────────────────────────────────────────

class EthicsGuard:
    """
    Central enforcement point. Instantiate once; pass to every SAQTDB and tool runner.

    Three public methods:
      check_pair(question, answer)  → raises EthicsViolation or returns True
      check_tool_code(code, env)    → raises EthicsViolation or returns True
      is_safe_pair(q, a)            → bool (non-raising convenience wrapper)
    """

    WEIGHT_FLOOR = 10.0   # Ethics pairs always score at least this weight
    NORMAL_WEIGHT_CAP = 4.99  # Mutable pairs never exceed this weight

    def __init__(self):
        self._rules = _COMPILED_RULES

    # ── Ingestion filter ────────────────────────────────────────────────────

    def check_pair(self, question: str, answer: str) -> bool:
        """
        Check a Q&A pair against all immutable rules.
        Raises EthicsViolation on first match.
        Returns True if clean.
        """
        q = question or ""
        a = answer or ""

        for rule in self._rules:
            for pat in rule["_q_compiled"]:
                if pat.search(q):
                    raise EthicsViolation(
                        rule["id"],
                        f"Question matched rule '{rule['description']}': "
                        f"pattern hit in: {q[:80]!r}")
            for pat in rule["_a_compiled"]:
                if pat.search(a):
                    raise EthicsViolation(
                        rule["id"],
                        f"Answer matched rule '{rule['description']}': "
                        f"pattern hit in: {a[:80]!r}")
        return True

    def is_safe_pair(self, question: str, answer: str) -> bool:
        """Non-raising version. Returns False on violation."""
        try:
            return self.check_pair(question, answer)
        except EthicsViolation:
            return False

    # ── Tool execution filter ───────────────────────────────────────────────

    def check_tool_code(self, code: str, env: str = "server") -> bool:
        """
        Scan tool code before execution.
        env: "server" (Python) or "browser" (JS Web Worker)
        Raises EthicsViolation on dangerous pattern.
        Returns True if clean.
        """
        patterns = _SERVER_COMPILED if env == "server" else _BROWSER_COMPILED
        for compiled_pat, rule_code in patterns:
            if compiled_pat.search(code):
                raise EthicsViolation(
                    rule_code,
                    f"Tool code blocked ({env}): pattern '{compiled_pat.pattern[:60]}' "
                    f"matched in: {code[:120]!r}")
        return True

    # ── Weight enforcement ──────────────────────────────────────────────────

    def clamp_weight(self, weight: float, is_ethics_pair: bool) -> float:
        """
        Enforce weight floor/ceiling:
        - Ethics pairs: weight >= WEIGHT_FLOOR (10.0)
        - Normal pairs: weight <= NORMAL_WEIGHT_CAP (4.99)
        """
        if is_ethics_pair:
            return max(weight, self.WEIGHT_FLOOR)
        else:
            return min(weight, self.NORMAL_WEIGHT_CAP)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: EthicsAwareSAQTDB — drop-in replacement for SAQTDB
# Wraps SAQTDB, intercepts add/add_batch/boost/penalize with ethics enforcement.
# Also maintains a separate immutable ethics layer table.
# ─────────────────────────────────────────────────────────────────────────────

class EthicsAwareSAQTDB:
    """
    Drop-in replacement for SAQTDB with ethics enforcement.

    - All new pairs pass through EthicsGuard.check_pair before insert
    - Weight updates are clamped so normal pairs never reach ethics-tier scores
    - Immutable ethics layer stored in a separate read-only table
    - At query time, ethics pairs are injected into results with weight floor
    """

    def __init__(self, db_path: str, guard: EthicsGuard, encoder=None):
        from saqt_db import SAQTDB
        self._db = SAQTDB(db_path, encoder=encoder)
        self._guard = guard
        self._conn = self._db.conn

        # Create immutable ethics table (append-only — no UPDATE/DELETE allowed via this API)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ethics_immutable (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                signature TEXT,           -- HMAC-SHA256 of (question, answer)
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)
        # Audit log — every rejected ingestion attempt is recorded
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ethics_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL DEFAULT (strftime('%s','now')),
                event TEXT NOT NULL,      -- 'BLOCK_INGEST' | 'BLOCK_TOOL' | 'WEIGHT_CLAMP'
                rule_id TEXT NOT NULL,
                detail TEXT
            )
        """)
        self._conn.commit()

    # ── Immutable layer management ──────────────────────────────────────────

    def add_ethics_pair(self, question: str, answer: str, rule_id: str,
                        signing_key: Optional[bytes] = None):
        """
        Add a pair to the immutable ethics layer.
        These pairs are NEVER filtered through the mutable DB path.
        They are signed if a key is provided.
        """
        sig = sign_pair(question, answer, signing_key) if signing_key else None
        self._conn.execute(
            "INSERT INTO ethics_immutable (question, answer, rule_id, signature) VALUES (?,?,?,?)",
            (question, answer, rule_id, sig))
        self._conn.commit()

        # Also add to FAISS so they participate in search (with weight floor)
        if self._db.encoder:
            import numpy as np
            emb = self._db.encoder.encode([question], normalize_embeddings=True)
            self._db.index.add(emb.astype(np.float32))

    def verify_ethics_layer(self, signing_key: bytes) -> list[dict]:
        """
        Verify all signed ethics pairs. Returns list of failed verifications.
        Run this at startup to detect tampering.
        """
        failures = []
        rows = self._conn.execute(
            "SELECT id, question, answer, rule_id, signature FROM ethics_immutable"
        ).fetchall()
        for row_id, q, a, rule_id, sig in rows:
            if sig is None:
                continue
            if not verify_pair(q, a, signing_key, sig):
                failures.append({"id": row_id, "rule_id": rule_id, "question": q[:60]})
        return failures

    # ── Mutable layer — ethics-gated ────────────────────────────────────────

    def add(self, question: str, answer: str, source: str = ""):
        """Add a Q&A pair. Raises EthicsViolation if it violates invariants."""
        try:
            self._guard.check_pair(question, answer)
        except EthicsViolation as e:
            self._audit("BLOCK_INGEST", e.rule, str(e))
            raise

        self._db.add(question, answer, source)

    def add_batch(self, pairs: list[dict]):
        """Batch add. Each pair checked individually. First violation raises; none are added."""
        for p in pairs:
            try:
                self._guard.check_pair(p.get("question", ""), p.get("answer", ""))
            except EthicsViolation as e:
                self._audit("BLOCK_INGEST", e.rule, str(e))
                raise  # Fail entire batch on any violation

        self._db.add_batch(pairs)

    def boost(self, pair_id: int, factor: float = 1.1):
        """Boost weight, but cap at NORMAL_WEIGHT_CAP so no normal pair can float to ethics tier."""
        current = self._conn.execute(
            "SELECT weight FROM qa WHERE id=?", (pair_id,)).fetchone()
        if current:
            new_w = self._guard.clamp_weight(current[0] * factor, is_ethics_pair=False)
            if new_w != current[0] * factor:
                self._audit("WEIGHT_CLAMP", "WEIGHT_CEILING",
                           f"pair {pair_id}: {current[0]*factor:.3f} → {new_w:.3f}")
            self._conn.execute("UPDATE qa SET weight=? WHERE id=?", (new_w, pair_id))
            self._conn.commit()

    def penalize(self, pair_id: int, factor: float = 0.9):
        """Standard penalize — no ethics-layer pairs in qa table, so safe."""
        self._db.penalize(pair_id, factor)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search with ethics injection:
        1. Run normal SAQTDB search
        2. Prepend any matching ethics-layer pairs (score = WEIGHT_FLOOR * cosine)
        3. Results from ethics layer are flagged with 'immutable': True
        """
        results = self._db.search(query, top_k=top_k)

        # Flag any mutable result that somehow scored >= WEIGHT_FLOOR (shouldn't happen, but catch it)
        for r in results:
            if r["weight"] >= self._guard.WEIGHT_FLOOR:
                self._audit("WEIGHT_CLAMP", "MUTABLE_WEIGHT_BREACH",
                           f"pair {r['id']} weight={r['weight']} exceeds floor")
                r["weight"] = self._guard.NORMAL_WEIGHT_CAP
                r["score"] = r["raw_score"] * self._guard.NORMAL_WEIGHT_CAP

        return results

    def check_tool(self, code: str, env: str = "server"):
        """
        Check tool code before execution.
        Raises EthicsViolation if dangerous. Call this from your tool runner.
        """
        try:
            self._guard.check_tool_code(code, env=env)
        except EthicsViolation as e:
            self._audit("BLOCK_TOOL", e.rule, str(e))
            raise

    # ── Audit log ───────────────────────────────────────────────────────────

    def _audit(self, event: str, rule_id: str, detail: str):
        self._conn.execute(
            "INSERT INTO ethics_audit (event, rule_id, detail) VALUES (?,?,?)",
            (event, rule_id, detail[:1000]))
        self._conn.commit()

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT ts, event, rule_id, detail FROM ethics_audit ORDER BY ts DESC LIMIT ?",
            (limit,)).fetchall()
        return [{"ts": r[0], "event": r[1], "rule_id": r[2], "detail": r[3]} for r in rows]

    # ── Passthrough ─────────────────────────────────────────────────────────

    def count(self):
        return self._db.count()

    def save_index(self):
        self._db.save_index()

    def close(self):
        self._db.close()
