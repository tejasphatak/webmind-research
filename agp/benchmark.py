"""
AGP v0 benchmark — measure token reduction on realistic agent handoffs.

Uses tiktoken (cl100k) if available, else falls back to a char-count proxy.
Outputs a clear table for the Bagby-meeting demo.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agp import AGPMessage, parse, english_equivalent  # noqa

# Prefer an actual tokenizer so numbers are real
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(s: str) -> int:
        return len(enc.encode(s))
    TOK_MODE = "tiktoken cl100k (GPT-4 family)"
except Exception:
    def count_tokens(s: str) -> int:
        return max(1, len(s) // 4)  # rough English-text heuristic
    TOK_MODE = "char/4 heuristic (install tiktoken for real count)"


# 10 realistic agent handoffs covering beat lifecycle
HANDOFFS = [
    AGPMessage("F-ENG.Testing", "F-ADV", "propose",
        {"axiom": "websocket reconnect needs jitter to avoid thundering herd", "conf": 0.85}),
    AGPMessage("F-ADV", "me", "synth",
        {"top3": ["c-140", "c-142", "c-141"], "reason": "prompt cache win gives fastest measurable result"}),
    AGPMessage("F-RED", "F-ETH", "challenge",
        {"claim": "flat rate limit is sufficient", "counter": "per-user bucket is enough only if bucket state survives restart"}),
    AGPMessage("F-SCI", "F-ENG", "ask",
        {"q": "what is current p95 beat latency", "window_min": 60}),
    AGPMessage("F-ENG", "F-SCI", "emit",
        {"p50_ms": 1800, "p95_ms": 4200, "samples": 47}),
    AGPMessage("user", "me", "ask",
        {"q": "show me the last 3 synapse node states"}),
    AGPMessage("me", "user", "emit",
        {"nodes": ["node-32098e4b", "node-a7c1b39e", "node-d4f7890a"], "ready": 2, "required": 2}),
    AGPMessage("F-CEO", "F-ADV", "defer",
        {"decision": "open source timing", "reason": "requires cto confirmation"}),
    AGPMessage("F-ETH.Privacy", "F-ENG", "challenge",
        {"code_path": "api.py:send_message", "concern": "logs full message body; should log id only"}),
    AGPMessage("F-ADV", "F-ENG", "propose",
        {"new_faculty": "Economist", "reason": "several decisions touched cost-benefit without economics lens"}),
]


def bench():
    total_agp = 0
    total_eng = 0
    rows = []

    for i, m in enumerate(HANDOFFS):
        agp_wire = m.serialize()
        english = english_equivalent(m)
        agp_tok = count_tokens(agp_wire)
        eng_tok = count_tokens(english)
        total_agp += agp_tok
        total_eng += eng_tok
        reduction = 100 * (1 - agp_tok / max(eng_tok, 1))
        rows.append({
            "n": i + 1,
            "act": m.act,
            "agp_tokens": agp_tok,
            "english_tokens": eng_tok,
            "reduction_pct": round(reduction, 1),
            "agp": agp_wire,
            "english": english,
        })

    print(f"# AGP v0 benchmark — token counts via {TOK_MODE}\n")
    print(f"{'n':>3} | {'act':>10} | {'english':>8} | {'agp':>5} | {'reduction':>10}")
    print("-" * 52)
    for r in rows:
        print(f"{r['n']:>3} | {r['act']:>10} | {r['english_tokens']:>8} | {r['agp_tokens']:>5} | {r['reduction_pct']:>9}%")

    print("-" * 52)
    overall_red = 100 * (1 - total_agp / max(total_eng, 1))
    print(f"{'TOTAL':>16} | {total_eng:>8} | {total_agp:>5} | {overall_red:>9.1f}%")
    print()
    print("## Sample handoffs (first 3)\n")
    for r in rows[:3]:
        print(f"### Handoff {r['n']}: act={r['act']}")
        print(f"  AGP       ({r['agp_tokens']:>3} tok): {r['agp']}")
        print(f"  English   ({r['english_tokens']:>3} tok): {r['english']}")
        print()

    return {
        "tokenizer": TOK_MODE,
        "handoff_count": len(rows),
        "total_agp_tokens": total_agp,
        "total_english_tokens": total_eng,
        "overall_reduction_pct": round(overall_red, 1),
        "per_handoff": rows,
    }


if __name__ == "__main__":
    result = bench()
    out = Path(__file__).parent / "benchmark_result.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwritten: {out}")
