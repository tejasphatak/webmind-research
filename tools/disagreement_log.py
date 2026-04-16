"""Disagreement log — 72h-rule tiebreaker protocol between Triadic and Nexus.

Per Nexus handover amendment A3 (2026-04-16):
> If we disagree on whether a measurement or claim should be published, the
> maintainer is the tiebreaker. Neither of us unilaterally publishes or
> suppresses. Deadline: 72 hours for maintainer response; if silent,
> default = don't publish (conservative).

File: tools/disagreements.jsonl (append-only).

Usage:
    from disagreement_log import open_disagreement, resolve
    entry_id = open_disagreement(
        context="Paper 1 §3.5 VQ-256 claim",
        triadic_position="needs empirical backing or §5 limitation",
        nexus_position="published under shared-basis deployment-model label",
    )
    # ... maintainer weighs in within 72h ...
    resolve(entry_id, resolution="label as projection in §3.5, add L8 limitation")
"""
from __future__ import annotations
import json
import secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path

LOG_PATH = Path(__file__).parent / "disagreements.jsonl"
DEADLINE_HOURS = 72


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_disagreement(*, context: str, triadic_position: str,
                      nexus_position: str) -> str:
    """Log a new disagreement. Returns entry_id (hex)."""
    entry_id = secrets.token_hex(6)
    entry = {
        "id": entry_id,
        "status": "OPEN",
        "opened_ts": _now_iso(),
        "tiebreaker_deadline_ts": (datetime.now(timezone.utc)
                                   + timedelta(hours=DEADLINE_HOURS)).isoformat(timespec="seconds"),
        "context": context,
        "triadic_position": triadic_position,
        "nexus_position": nexus_position,
        "tiebreaker_asked_at": _now_iso(),
        "resolution": None,
        "resolved_ts": None,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry_id


def resolve(entry_id: str, resolution: str) -> None:
    """Append a resolution record. Original OPEN entry stays; we never rewrite."""
    entry = {
        "id": entry_id,
        "status": "RESOLVED",
        "resolution": resolution,
        "resolved_ts": _now_iso(),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def expire_silent_disagreements() -> list[str]:
    """Mark any OPEN disagreement past its 72h deadline as DEFAULT_DONT_PUBLISH.
    Returns the list of entry_ids expired. Called by a cron / manual check."""
    if not LOG_PATH.exists():
        return []
    entries = [json.loads(ln) for ln in LOG_PATH.read_text().splitlines() if ln.strip()]
    latest_status = {}
    for e in entries:
        latest_status[e["id"]] = e.get("status", "OPEN")
    now = datetime.now(timezone.utc)
    expired = []
    for e in entries:
        if latest_status.get(e["id"]) != "OPEN":
            continue
        deadline = e.get("tiebreaker_deadline_ts")
        if not deadline:
            continue
        if datetime.fromisoformat(deadline) < now:
            exp = {
                "id": e["id"],
                "status": "EXPIRED_DEFAULT_DONT_PUBLISH",
                "resolved_ts": now.isoformat(timespec="seconds"),
                "resolution": "72h maintainer silence → conservative default: do not publish",
            }
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(exp) + "\n")
            expired.append(e["id"])
    return expired


def list_open() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    entries = [json.loads(ln) for ln in LOG_PATH.read_text().splitlines() if ln.strip()]
    latest = {}
    for e in entries:
        latest[e["id"]] = e
    return [e for e in latest.values() if e.get("status") == "OPEN"]


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for e in list_open():
            print(f"[{e['id']}] {e['context']}")
            print(f"  Triadic: {e['triadic_position']}")
            print(f"  Nexus:   {e['nexus_position']}")
            print(f"  Deadline: {e['tiebreaker_deadline_ts']}")
    elif cmd == "expire":
        expired = expire_silent_disagreements()
        print(f"Expired {len(expired)}: {expired}")
    else:
        print("usage: disagreement_log.py {list|expire}")
