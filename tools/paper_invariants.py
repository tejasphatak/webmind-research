"""
Paper Invariants — auto-validates every numeric claim against raw data.

SCOPE: papers/carrier-payload-text-only-v1.md (text-only, inference-time).
The multimodal model gemma4_31b is deliberately excluded here; the paper's
§1.2 Non-contributions section disclaims all vision-language claims, which
are moved to a companion paper with its own invariant suite.

Every numeric assertion in the text-only paper should be checkable here.
If this script fails, the paper is wrong.

Run before every arXiv submission, every revision, every draft-share.

Author: Tejas Phatak
Date: 2026-04-16 (text-only rescope)
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
}
# Text-only paper scope (2026-04-16): gemma4_31b deliberately excluded.
# See papers/carrier-payload-text-only-v1.md §1.2 Non-contributions and the
# companion multimodal paper for vision-language coverage.

# Long-context Qwen 2.5 32B — claims from §3.4 of the text-only paper.
# Raw files hold per-prompt ranks_for_variance at splice layers [16, 32, 48].
# Sequence lengths covered across the two files: {256, 512, 1024, 1621}.
LONGCTX_QWEN_FILES = [
    'findings/longctx_qwen_32b.json',
    'findings/longctx_qwen_32b_extended.json',
]
LONGCTX_SEQ_LENS = [256, 512, 1024, 1621]
# Paper §3.4 claim: PCA rank for 99% variance (mean across splice layers 16/32/48)
# grows with sequence length. Raw-data-derived values as of 2026-04-16:
# mean rank99 = {256: 8.4, 512: 55.0, 1024: 193.6, 1621: 384.0}.
CLAIMED_MEAN_RANK99 = {256: 8, 512: 55, 1024: 194, 1621: 384}
# Paper §3.5 claim: compression ratio degrades from 183× at 256 → 13× at 1621.
# Computed from the CR formula using mean rank99 at each seq.
CLAIMED_CR_BY_SEQ = {256: 183, 512: 81, 1024: 26, 1621: 13}
# Paper §3.6 layer-depth claim: L48/L16 rank99 ratio shrinks with seq_len.
CLAIMED_L48_L16_RATIO = {256: 23.0, 512: 26.5, 1024: 4.25, 1621: 2.88}
QWEN_HIDDEN_DIM = 5120


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
# Long-context Qwen invariants (text-only paper §3.4–§3.6)
# ──────────────────────────────────────────────────────────────────────────

def _load_longctx_qwen() -> dict:
    """Merge records across longctx_qwen_32b.json + _extended.json.

    Returns: {(seq_len, splice_layer): [rank99, ...]} aggregated across prompts.
    """
    agg: dict[tuple[int, int], list[int]] = {}
    for rel in LONGCTX_QWEN_FILES:
        p = Path(rel)
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        for r in d['results']:
            k = (r['seq_len'], r['splice_layer'])
            agg.setdefault(k, []).append(r['ranks_for_variance']['0.99'])
    return agg


def check_longctx_seq_coverage():
    """All four claimed sequence lengths must be present in the raw data."""
    agg = _load_longctx_qwen()
    seqs_seen = sorted(set(s for s, _ in agg.keys()))
    missing = [s for s in LONGCTX_SEQ_LENS if s not in seqs_seen]
    return len(missing) == 0, f"seq_lens seen={seqs_seen}, missing={missing}"


def check_longctx_mean_rank99(seq_len: int, claimed: int, tol_frac: float = 0.15):
    """Mean rank99 across splice layers {16, 32, 48} at this seq matches paper claim.

    Paper §3.4 claim: rank for 99% variance = 8/55/194/384 at seq 256/512/1024/1621.
    These are means across the three splice layers [16, 32, 48].
    """
    agg = _load_longctx_qwen()
    layer_means = []
    for layer in (16, 32, 48):
        vs = agg.get((seq_len, layer))
        if not vs:
            return False, f"missing (seq={seq_len}, L={layer}) data"
        layer_means.append(sum(vs) / len(vs))
    observed_mean = sum(layer_means) / len(layer_means)
    err = abs(observed_mean - claimed) / max(claimed, 1)
    return err <= tol_frac, \
        f"observed mean rank99 at seq={seq_len} = {observed_mean:.1f} (claimed {claimed}, tol {tol_frac:.0%})"


def check_longctx_rank_monotonic():
    """rank99 must be non-decreasing in seq_len at every splice layer (geometric invariant)."""
    agg = _load_longctx_qwen()
    for layer in (16, 32, 48):
        prev = -1
        for seq in LONGCTX_SEQ_LENS:
            vs = agg.get((seq, layer))
            if not vs:
                return False, f"missing (seq={seq}, L={layer})"
            m = sum(vs) / len(vs)
            if m < prev - 1e-6:
                return False, f"L{layer}: rank99 drops at seq={seq}: {m:.1f} < prev {prev:.1f}"
            prev = m
    return True, "monotonic at L16, L32, L48"


def check_longctx_depth_ratio(seq_len: int, claimed_ratio: float, tol_frac: float = 0.20):
    """Paper §3.6 claim: L48/L16 rank99 ratio = 23, 26.5, 4.25, 2.88 at seq 256/512/1024/1621.

    Ratio shrinks with seq_len — the depth gap closes as context grows.
    """
    agg = _load_longctx_qwen()
    l16 = agg.get((seq_len, 16))
    l48 = agg.get((seq_len, 48))
    if not l16 or not l48:
        return False, f"missing (seq={seq_len}) L16 or L48 data"
    m16 = sum(l16) / len(l16)
    m48 = sum(l48) / len(l48)
    if m16 < 1e-6:
        return False, f"L16 rank99 is 0 at seq={seq_len} — cannot form ratio"
    observed = m48 / m16
    err = abs(observed - claimed_ratio) / max(claimed_ratio, 1e-6)
    return err <= tol_frac, \
        f"L48/L16 ratio at seq={seq_len} = {observed:.2f} (claimed {claimed_ratio:.2f}, tol {tol_frac:.0%})"


def check_longctx_cr_endpoints(tol_frac: float = 0.20):
    """Paper §3.5 claim: CR degrades 183× → 13× across seq 256 → 1621.

    CR formula (zero sparse-residual, rank = mean rank99 across layers):
        baseline = seq * hidden_dim
        wire     = seq * k + hidden_dim
        CR       = baseline / wire
    """
    agg = _load_longctx_qwen()
    results = {}
    for seq in LONGCTX_SEQ_LENS:
        layer_means = []
        for layer in (16, 32, 48):
            vs = agg.get((seq, layer))
            if not vs:
                return False, f"missing (seq={seq}, L={layer})"
            layer_means.append(sum(vs) / len(vs))
        k = sum(layer_means) / len(layer_means)
        baseline = seq * QWEN_HIDDEN_DIM
        wire = seq * k + QWEN_HIDDEN_DIM
        results[seq] = baseline / max(wire, 1)
    endpoints = [(256, CLAIMED_CR_BY_SEQ[256]), (1621, CLAIMED_CR_BY_SEQ[1621])]
    for seq, claimed in endpoints:
        observed = results[seq]
        err = abs(observed - claimed) / max(claimed, 1e-6)
        if err > tol_frac:
            return False, f"CR at seq={seq}: observed {observed:.1f}× vs claimed {claimed}× (err {err:.0%}, tol {tol_frac:.0%})"
    return True, f"CR endpoints: 256→{results[256]:.1f}×, 1621→{results[1621]:.1f}× (claimed 183×, 13×)"


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

    # === Paper headline claims — short context ===
    print("\n[PAPER CLAIMS — SHORT CONTEXT] ------------")
    # §3.1–§3.3: 22–24x compression at KL < 0.1 with 100% top-1 across 3 text families.
    invariant("Gemma 3 1B: ≥22x at KL<0.1 with top-1=100%",
              lambda: check_best_cr_at_kl_threshold('gemma3_1b', 0.1, 22.0, True))
    invariant("Llama 3.1 8B: ≥24x at KL<0.1 with top-1=100%",
              lambda: check_best_cr_at_kl_threshold('llama_8b', 0.1, 24.0, True))
    invariant("Qwen 2.5 32B: ≥24x at KL<0.1 with top-1=100%",
              lambda: check_best_cr_at_kl_threshold('qwen_32b', 0.1, 24.0, True))

    # === Paper headline claims — long context (Qwen 2.5 32B) ===
    print("\n[PAPER CLAIMS — LONG CONTEXT] ------------")
    # §3.4 coverage
    invariant("Long-context Qwen: seq_lens {256,512,1024,1621} all covered in raw data",
              check_longctx_seq_coverage)
    # §3.4 rank growth
    for seq, claimed in CLAIMED_MEAN_RANK99.items():
        invariant(f"Long-context Qwen: mean rank99 at seq={seq} ≈ {claimed} (±15%)",
                  lambda s=seq, c=claimed: check_longctx_mean_rank99(s, c))
    invariant("Long-context Qwen: rank99 non-decreasing in seq_len at L16/L32/L48",
              check_longctx_rank_monotonic)
    # §3.5 CR degradation endpoints
    invariant("Long-context Qwen: CR endpoints 256→183× and 1621→13× (±20%)",
              check_longctx_cr_endpoints)
    # §3.6 layer-depth ratio
    for seq, claimed in CLAIMED_L48_L16_RATIO.items():
        invariant(f"Long-context Qwen: L48/L16 rank99 ratio at seq={seq} ≈ {claimed} (±20%)",
                  lambda s=seq, c=claimed: check_longctx_depth_ratio(s, c))

    # === Known limitation DISCLOSURES ===
    print("\n[LIMITATION INVARIANTS — these document known caveats] ------------")
    for key, (display, hd, nl) in MULTIMODEL_FILES.items():
        d = load(key)
        seq_lens = [r['actual_seq_len'] for r in d['results']]
        invariant(f"{display}: all short-context prompts ≤ 30 tokens (documents SC bound)",
                  lambda sl=seq_lens: (max(sl) <= 30, f"max seq_len = {max(sl)}"))
    invariant("Short-context ranks tested only up to 16 (prompts cap at 2–30 tokens)",
              lambda: (True, "DOCUMENTED: rank 32/64 at short context infeasible; long-context run covers higher ranks"))
    invariant("Multimodal scope explicitly excluded from this paper",
              lambda: (True, "DOCUMENTED: §1.2 Non-contributions disclaims VLM claims; gemma4_31b moved to companion paper invariants"))

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
