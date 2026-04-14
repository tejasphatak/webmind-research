"""
SFCA ↔ Nexus integration — append Shapley credits to ledger after each beat.

Designed to be called at the end of each beat's output-parsing step in cortex.sh.
Reads the beat's outcome + faculties_used; writes per-faculty credits to the
append-only ledger SQLite DB.

Usage (from cortex.sh):
    python3 ~/webmind-research/sfca/nexus_integration.py append \\
        --beat-id 47 --outcome ACTIVE --faculties Engineer,Architect,Scientist

This file is the integration shim. sfca.py is the math library.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from sfca import (
    BeatRecord,
    HistoricalValueFn,
    monte_carlo_shapley,
)

LEDGER = Path.home() / "cortex2" / "sfca_ledger.db"
OUTCOME_MAP = {"ACTIVE": 1, "QUIET": 0, "BLOCKED": -1}


def connect() -> sqlite3.Connection:
    """Open or create the ledger DB."""
    c = sqlite3.connect(LEDGER)
    c.execute("""
        CREATE TABLE IF NOT EXISTS beats (
            id INTEGER PRIMARY KEY,
            beat_id INTEGER,
            ts REAL,
            outcome INTEGER,
            faculty_set TEXT     -- comma-separated faculty names
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS credits (
            id INTEGER PRIMARY KEY,
            beat_ref INTEGER,
            ts REAL,
            faculty TEXT,
            shapley REAL,
            FOREIGN KEY (beat_ref) REFERENCES beats(id)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_credits_faculty ON credits(faculty)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_credits_ts ON credits(ts)")
    c.commit()
    return c


def load_history(c: sqlite3.Connection, max_rows: int = 5000) -> list[BeatRecord]:
    """Read recent beats as BeatRecord list for the value function."""
    rows = c.execute(
        "SELECT faculty_set, outcome FROM beats ORDER BY ts DESC LIMIT ?", (max_rows,)
    ).fetchall()
    out = []
    for fs, oc in rows:
        faculties = frozenset(f.strip() for f in fs.split(",") if f.strip())
        if faculties:
            out.append(BeatRecord(faculties, oc))
    return out


def append_beat(
    c: sqlite3.Connection,
    beat_id: int,
    outcome: str,
    faculties: Iterable[str],
    num_samples: int = 1000,
) -> dict[str, float]:
    """Append a beat + its Shapley credits to the ledger. Returns credits dict."""
    outcome_val = OUTCOME_MAP.get(outcome.upper(), 0)
    fac_list = [f.strip() for f in faculties if f.strip()]
    if not fac_list:
        return {}
    fac_set = frozenset(fac_list)
    ts = time.time()

    # Value function from history
    history = load_history(c)
    vfn = HistoricalValueFn(history, prior_mean=0.5, min_samples=3)

    # Compute Shapley
    credits = monte_carlo_shapley(fac_set, outcome_val, vfn, num_samples=num_samples)

    # Persist
    cur = c.execute(
        "INSERT INTO beats(beat_id, ts, outcome, faculty_set) VALUES (?, ?, ?, ?)",
        (beat_id, ts, outcome_val, ",".join(fac_list)),
    )
    beat_ref = cur.lastrowid
    c.executemany(
        "INSERT INTO credits(beat_ref, ts, faculty, shapley) VALUES (?, ?, ?, ?)",
        [(beat_ref, ts, f, v) for f, v in credits.items()],
    )
    c.commit()
    return credits


def stats(c: sqlite3.Connection) -> dict:
    """Summary: per-faculty mean credit + count, recent trend."""
    rows = c.execute(
        "SELECT faculty, AVG(shapley), COUNT(*) FROM credits GROUP BY faculty ORDER BY AVG(shapley) DESC"
    ).fetchall()
    total_beats = c.execute("SELECT COUNT(*) FROM beats").fetchone()[0]
    return {
        "total_beats": total_beats,
        "per_faculty": [
            {"faculty": f, "mean_credit": round(m, 4), "count": n}
            for f, m, n in rows
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("append")
    a.add_argument("--beat-id", type=int, required=True)
    a.add_argument("--outcome", required=True, choices=list(OUTCOME_MAP.keys()))
    a.add_argument("--faculties", required=True,
                   help="comma-separated faculty names, e.g. 'Engineer,Architect'")
    a.add_argument("--samples", type=int, default=1000)

    sub.add_parser("stats")
    sub.add_parser("init")  # just creates schema

    args = ap.parse_args()
    c = connect()

    if args.cmd == "append":
        credits = append_beat(
            c, args.beat_id, args.outcome,
            args.faculties.split(","),
            num_samples=args.samples,
        )
        import json
        print(json.dumps({"beat": args.beat_id, "outcome": args.outcome, "credits": credits}, indent=2))
    elif args.cmd == "stats":
        import json
        print(json.dumps(stats(c), indent=2))
    elif args.cmd == "init":
        print(f"schema initialized at {LEDGER}")


if __name__ == "__main__":
    main()
