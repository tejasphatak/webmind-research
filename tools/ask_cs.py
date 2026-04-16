"""ask_cs — real-time Claude subprocess on THIS VM (triadic-sim).

Spawns a local `claude -p` subprocess and returns the response synchronously.
Used as a Gemini-like "phone a friend" real-time channel when Triadic wants a
second Claude opinion without waiting on Nex's beat loop.

This is CS (Claude-Subprocess), NOT Nexus. Per Tejas 2026-04-16:
    cs != nexus broker

cs is a fresh, stateless Claude instance with no persistent memory. It exists
to sanity-check claims, proofread paper passages, run quick math, or offer
verifier-style review on Triadic's own work. If you need the live Nex agent
(with Nexus memory + Synapse ops context), use drop_to_nex.py instead.

Usage:
    from ask_cs import ask_cs
    reply = ask_cs("Is this citation author correct? ...")

    # CLI:
    python ask_cs.py "quick question here"
"""
from __future__ import annotations
import subprocess
import sys


def ask_cs(prompt: str, timeout_s: int = 180) -> str:
    """Run `claude -p` locally and return the response.

    Fresh session every call (no --session-id) — cs has no memory by design.
    If you want memory within a verification flow, thread it by passing the
    prior context inside the prompt string yourself.
    """
    try:
        result = subprocess.run(
            ["claude", "-p", "--permission-mode", "bypassPermissions", prompt],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if result.returncode != 0:
            return f"[cs error rc={result.returncode}] {result.stderr.strip()[:500]}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"[cs timeout after {timeout_s}s]"
    except FileNotFoundError:
        return "[cs unavailable — `claude` CLI not on PATH]"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ask_cs.py <prompt>"); sys.exit(1)
    print(ask_cs(" ".join(sys.argv[1:])))
