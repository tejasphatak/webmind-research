"""Gate-log — append-only audit record of every gate decision.

Per Nexus handover amendment A1 (2026-04-16):
> Gate-13 visibility, not veto. Gate decisions logged somewhere I can read.
> Not to veto; to notice if a gate is ever used to suppress rather than validate.

Every validator should call `record(paper, claim, decision, reason, validators_run)`
exactly once per run. File lives at webmind-research/tools/gate-log.jsonl —
append-only, never rewritten.

Readers (Nex or maintainer) can tail this to audit what was gated, why, and
by which validator. This turns gating from a black-box into a transparent record.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).parent / "gate-log.jsonl"


def record(*, paper: str, claim: str, decision: str, reason: str,
           validators_run: list[str], gate: str | None = None) -> None:
    """Append one gate decision to the log.

    Args:
        paper: paper filename (e.g. 'carrier-payload-text-only-v1.md')
        claim: what was checked (e.g. 'all citations primary-source verified')
        decision: 'PASS', 'FAIL', 'SKIP', 'WAIVED'
        reason: one-line human-readable explanation
        validators_run: list of tool/invariant IDs that contributed
        gate: SUBMISSION_GATING.md gate number (e.g. 'G2'). Optional.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper": paper,
        "gate": gate,
        "claim": claim,
        "decision": decision,
        "reason": reason,
        "validators_run": validators_run,
        "author": os.environ.get("GATE_AUTHOR", "triadic"),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def tail(n: int = 20) -> list[dict]:
    """Return the last n entries (human inspection / Nex audit)."""
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return [json.loads(ln) for ln in lines[-n:]]


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "tail":
        n = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
        for e in tail(n):
            print(f"[{e['ts']}] {e.get('gate','?')} {e['paper']} → {e['decision']}: {e['claim']} ({e['reason']})")
    else:
        print(__doc__)
