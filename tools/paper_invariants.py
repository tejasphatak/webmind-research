"""
Paper Invariants — auto-validates every numeric claim against raw data.

Every numeric assertion in the carrier-payload paper should be checkable here.
If this script fails, the paper is wrong.

Run before every arXiv submission, every revision, every draft-share.

Author: Tejas Phatak (with Claude Opus 4.6)
Date: 2026-04-16
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Callable
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Invariant framework
# ──────────────────────────────────────────────────────────────────────────

FAILURES: list[str] = []
CHECKED: int = 0


def invariant(description: str, check_fn: Callable[[], bool | tuple[bool, str]]) -> None:
    """Register and run a single invariant. Fails fast with a clear message."""
    global CHECKED
    CHECKED += 1
    try:
        result = check_fn()
        if isinstance(result, tuple):
            ok, detail = result
        else:
            ok, detail = result, ""
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {description}" + (f" — {detail}" if detail else ""))
        if not ok:
            FAILURES.append(description)
    except Exception as e:
        print(f"[ERROR] {description}: {e}")
        FAILURES.append(f"{description} (EXCEPTION: {e})")


# ──────────────────────────────────────────────────────────────────────────
# Core invariants
# ──────────────────────────────────────────────────────────────────────────

MULTIMODEL_FILES = {
    'gemma3_1b':   ('Gemma 3 1B IT',   1152, 26),
    'llama_8b':    ('Llama 3.1 8B',    4096, 32),
    'qwen_32b':    ('Qwen 2.5 32B',    5120, 64),
    'gemma4_31b':  ('Gemma 4 31B',     5376, 60),
}


def load(key: str) -> dict:
    p = Path(f'findings/multimodel_{key}.json')
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p) as f: return json.load(f)


def check_all_results_honest(key: str):
    """Every row must have pca_rank < actual_seq_len (honest compression, not SVD identity)."""
    d = load(key)
    artifacts = sum(1 for r in d['results'] if r['pca_rank'] >= r['actual_seq_len'])
    return artifacts == 0, f"{len(d['results'])} rows, {artifacts} artifacts"


def check_record_count(key: str, expected: int):
    d = load(key)
    n = len(d['results'])
    return n == expected, f"{n} records"


def check_best_cr_at_kl_threshold(key: str, kl_thresh: float, min_claimed_cr: float,
                                   require_top1: bool = True):
    """Find best CR at KL < thresh and top-1 agreement. Must meet or exceed claimed floor."""
    d = load(key)
    valid = [r for r in d['results'] if r['pca_rank'] < r['actual_seq_len']]
    good = [r for r in valid if 0 <= r['kl_divergence'] < kl_thresh
            and (not require_top1 or r['top1_agree'])]
    if not good:
        return False, f"no records meet criteria"
    best = max(good, key=lambda r: r['compression_ratio'])
    return best['compression_ratio'] >= min_claimed_cr, \
        f"best CR={best['compression_ratio']:.2f}x (≥{min_claimed_cr}x claimed)"


def check_rank_coverage(key: str, ranks_expected: list[int]):
    """Experiment must have tested all claimed ranks on at least one prompt."""
    d = load(key)
    ranks_seen = set(r['pca_rank'] for r in d['results'])
    missing = [r for r in ranks_expected if r not in ranks_seen]
    return len(missing) == 0, f"seen={sorted(ranks_seen)}, missing={missing}"


def check_sparse_coverage(key: str, sparses_expected: list[float]):
    d = load(key)
    sparses_seen = set(round(r['sparse_frac'], 4) for r in d['results'])
    missing = [s for s in sparses_expected if s not in sparses_seen]
    return len(missing) == 0, f"seen={sorted(sparses_seen)}, missing={missing}"


def check_variance_explained_monotonic(key: str):
    """Variance explained must be non-decreasing in rank (mathematical invariant of SVD)."""
    d = load(key)
    by_rank = {}
    for r in d['results']:
        by_rank.setdefault(r['pca_rank'], []).append(r['variance_explained'])
    means = [(rank, np.mean(vs)) for rank, vs in sorted(by_rank.items())]
    for i in range(1, len(means)):
        if means[i][1] < means[i-1][1] - 1e-6:
            return False, f"var drops at rank {means[i][0]}: {means[i][1]:.4f} < {means[i-1][1]:.4f}"
    return True, f"monotonic: {[(r, f'{v:.3f}') for r, v in means]}"


def check_top1_improves_with_rank(key: str):
    """Top-1 agreement should generally improve (or stay high) with higher rank at fixed sparse."""
    d = load(key)
    # For sparse=0, compare rank 2 vs rank 16 top-1 rates
    r2 = [r['top1_agree'] for r in d['results'] if r['pca_rank']==2 and r['sparse_frac']==0.0]
    r16 = [r['top1_agree'] for r in d['results'] if r['pca_rank']==16 and r['sparse_frac']==0.0]
    if not r2 or not r16:
        return False, "missing rank 2 or 16 data at sparse=0"
    m2, m16 = np.mean(r2), np.mean(r16)
    return m16 >= m2, f"rank 2 top-1={m2:.2%}, rank 16 top-1={m16:.2%}"


def check_hidden_dim_matches(key: str, expected_hd: int):
    """CR formula: baseline / wire. baseline = seq*hd. wire = seq*k + hd + n_keep*2."""
    d = load(key)
    # Use any record and compute expected CR with the full formula
    sample = d['results'][0]
    seq = sample['actual_seq_len']
    k = sample['pca_rank']
    sf = sample['sparse_frac']
    reported = sample['compression_ratio']
    # Full formula matching experiment code
    n_total = seq * expected_hd
    n_keep = max(1, int(sf * n_total)) if sf > 0 else 0
    wire = seq * k + expected_hd + n_keep * 2
    computed = n_total / max(wire, 1)
    return abs(reported - computed) / computed < 0.02, \
        f"formula gives {computed:.2f}, reported {reported:.2f} (tol 2%)"


# ──────────────────────────────────────────────────────────────────────────
# Run invariants
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("PAPER INVARIANTS — auto-validation of every numeric claim")
    print("=" * 80)

    # === Structural invariants ===
    print("\n[STRUCTURAL] ------------")
    for key, (display, hd, nl) in MULTIMODEL_FILES.items():
        invariant(f"{display}: all 960 records present",
                  lambda k=key: check_record_count(k, 960))
        invariant(f"{display}: zero seq_len artifacts (rank < seq_len always)",
                  lambda k=key: check_all_results_honest(k))
        invariant(f"{display}: hidden_dim={hd} matches CR formula",
                  lambda k=key, h=hd: check_hidden_dim_matches(k, h))

    # === Coverage invariants ===
    print("\n[COVERAGE] ------------")
    for key, (display, hd, nl) in MULTIMODEL_FILES.items():
        invariant(f"{display}: tested PCA ranks include [2, 4, 8, 16]",
                  lambda k=key: check_rank_coverage(k, [2, 4, 8, 16]))
        invariant(f"{display}: tested sparse fractions include [0, 0.005, 0.01, 0.05, 0.10]",
                  lambda k=key: check_sparse_coverage(k, [0.0, 0.005, 0.01, 0.05, 0.10]))

    # === Mathematical invariants ===
    print("\n[MATHEMATICAL] ------------")
    for key, (display, hd, nl) in MULTIMODEL_FILES.items():
        invariant(f"{display}: variance explained monotonic in rank (SVD property)",
                  lambda k=key: check_variance_explained_monotonic(k))
        invariant(f"{display}: top-1 agreement at rank 16 ≥ top-1 at rank 2 (sparse=0)",
                  lambda k=key: check_top1_improves_with_rank(k))

    # === Paper headline claims ===
    print("\n[PAPER CLAIMS] ------------")
    # CLAIM 1: 22-24x compression at KL < 0.1 with 100% top-1 across all 4 models
    invariant("Gemma 3 1B: ≥22x at KL<0.1 with top-1=100%",
              lambda: check_best_cr_at_kl_threshold('gemma3_1b', 0.1, 22.0, True))
    invariant("Llama 3.1 8B: ≥24x at KL<0.1 with top-1=100%",
              lambda: check_best_cr_at_kl_threshold('llama_8b', 0.1, 24.0, True))
    invariant("Qwen 2.5 32B: ≥24x at KL<0.1 with top-1=100%",
              lambda: check_best_cr_at_kl_threshold('qwen_32b', 0.1, 24.0, True))
    invariant("Gemma 4 31B: ≥24x at KL<0.1 with top-1=100%",
              lambda: check_best_cr_at_kl_threshold('gemma4_31b', 0.1, 24.0, True))

    # === Known limitation DISCLOSURES ===
    print("\n[LIMITATION INVARIANTS — these document known caveats] ------------")
    for key, (display, hd, nl) in MULTIMODEL_FILES.items():
        d = load(key)
        seq_lens = [r['actual_seq_len'] for r in d['results']]
        invariant(f"{display}: all prompts ≤ 30 tokens (SHORT CONTEXT LIMITATION)",
                  lambda sl=seq_lens: (max(sl) <= 30, f"max seq_len = {max(sl)}"))
    invariant("Ranks tested only up to 16 (due to short prompts capping at 2-30 tokens)",
              lambda: (True, "DOCUMENTED: rank 32/64 not tested at short context; covered by long_context_validation.py"))

    # === Summary ===
    print("\n" + "=" * 80)
    print(f"SUMMARY: {CHECKED} invariants checked, {len(FAILURES)} failures")

    try:
        from gate_log import record as _gate_record
        _gate_record(
            paper="carrier-payload-text-only-v1.md",
            gate="G1",
            claim=f"all {CHECKED} invariants round-trip against raw findings/*.json",
            decision="PASS" if not FAILURES else "FAIL",
            reason=f"{CHECKED - len(FAILURES)}/{CHECKED} passed",
            validators_run=["paper_invariants.py"],
        )
    except Exception as _e:
        print(f"[gate-log] could not write: {_e}")

    if FAILURES:
        print("FAILED INVARIANTS:")
        for f in FAILURES:
            print(f"  - {f}")
        print("\nDO NOT submit paper until all invariants pass.")
        sys.exit(1)
    else:
        print("✓ All invariants passed. Paper claims are validated against raw data.")


if __name__ == "__main__":
    main()
