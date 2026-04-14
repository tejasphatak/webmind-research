"""
AGP v0 — Agent Grammar Protocol.

A compact encoding for agent-to-agent handoffs in multi-persona LLM systems.
Goal: replace verbose English messages between agents/faculties with tight,
structured tokens while preserving semantic content.

v0 grammar (intentionally minimal):

  message := source ">" target ":" act " " data?
  source, target := ID (faculty, mode, role — see registry)
  act := one of { think, propose, challenge, synth, emit, ask, defer }
  data := key:value pairs separated by ','

  Examples:
    F-ENG.test>F-ADV:propose axiom:"ws-reconnect needs jitter",conf:0.8
    F-ADV>me:synth top3:c-140|c-142|c-141,reason:"cache win fastest"
    user>me:ask q:"synapse node count?"
    me>user:emit val:3,delta:+2

This file is the reference parser + serializer + token-count benchmark.
Licensed MIT under Webmind umbrella.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional


VALID_ACTS = {"think", "propose", "challenge", "synth", "emit", "ask", "defer"}


@dataclass
class AGPMessage:
    source: str
    target: str
    act: str
    data: dict[str, str | int | float | list] = field(default_factory=dict)

    def serialize(self) -> str:
        """Encode to AGP wire form."""
        parts = [f"{self.source}>{self.target}:{self.act}"]
        if self.data:
            kvs = []
            for k, v in self.data.items():
                kvs.append(f"{k}:{_encode_value(v)}")
            parts.append(" " + ",".join(kvs))
        return "".join(parts)


def _encode_value(v) -> str:
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "|".join(str(x) for x in v)
    s = str(v)
    # Quote if contains reserved chars
    if any(c in s for c in ',|":'):
        return '"' + s.replace('"', '\\"') + '"'
    return s


_MSG_RE = re.compile(
    r"^(?P<source>[A-Za-z0-9\-._]+)>(?P<target>[A-Za-z0-9\-._]+):"
    r"(?P<act>[a-z]+)(?:\s+(?P<data>.+))?$"
)


def parse(raw: str) -> AGPMessage:
    raw = raw.strip()
    m = _MSG_RE.match(raw)
    if not m:
        raise ValueError(f"malformed AGP message: {raw!r}")
    act = m.group("act")
    if act not in VALID_ACTS:
        raise ValueError(f"unknown act {act!r}, expected one of {VALID_ACTS}")
    data_str = m.group("data") or ""
    data = _parse_data(data_str) if data_str else {}
    return AGPMessage(
        source=m.group("source"),
        target=m.group("target"),
        act=act,
        data=data,
    )


def _parse_data(s: str) -> dict:
    out: dict = {}
    # simple tokenizer that respects quoted strings
    pos = 0
    n = len(s)
    while pos < n:
        # key
        colon = s.find(":", pos)
        if colon < 0:
            break
        key = s[pos:colon].strip()
        pos = colon + 1
        # value
        if pos < n and s[pos] == '"':
            # quoted
            end = pos + 1
            while end < n and s[end] != '"':
                if s[end] == "\\" and end + 1 < n:
                    end += 2
                    continue
                end += 1
            val = s[pos + 1:end].replace('\\"', '"')
            pos = end + 1
            if pos < n and s[pos] == ",":
                pos += 1
        else:
            comma = s.find(",", pos)
            end = comma if comma >= 0 else n
            val_raw = s[pos:end].strip()
            # typed parse
            if "|" in val_raw:
                val = val_raw.split("|")
            elif val_raw.lower() in ("true", "false"):
                val = val_raw.lower() == "true"
            else:
                try:
                    val = int(val_raw)
                except ValueError:
                    try:
                        val = float(val_raw)
                    except ValueError:
                        val = val_raw
            pos = end + 1 if comma >= 0 else n
        out[key] = val
    return out


def english_equivalent(msg: AGPMessage, registry: Optional[dict] = None) -> str:
    """Render an AGP message as the verbose English it replaces."""
    src = _resolve(msg.source, registry)
    tgt = _resolve(msg.target, registry)
    verbs = {
        "think": "is thinking and wants",
        "propose": "proposes to",
        "challenge": "is challenging",
        "synth": "synthesizes for",
        "emit": "emits to",
        "ask": "asks",
        "defer": "defers to",
    }
    parts = [f"{src} {verbs.get(msg.act, msg.act)} {tgt}."]
    for k, v in msg.data.items():
        if isinstance(v, list):
            v_str = ", ".join(str(x) for x in v)
        else:
            v_str = str(v)
        parts.append(f"The {k.replace('_', ' ')} is {v_str!r}.")
    return " ".join(parts)


def _resolve(idd: str, registry: Optional[dict]) -> str:
    if not registry or idd not in registry.get("entries", {}):
        return idd
    e = registry["entries"][idd]
    return e.get("codename", idd)
