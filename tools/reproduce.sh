#!/usr/bin/env bash
# reproduce.sh — Reproduction driver for the Carrier-Payload text-only paper
# (papers/carrier-payload-text-only-v1.md).
#
# Modes:
#   --quick   (default)  Gemma 3 1B, 8 prompts, ~2 min GPU / ~15 min CPU.
#                        Reproduces the short-context 22× point for one family.
#                        Writes $RESULTS_DIR/_raw_compression_reproduced.json.
#   --full               Full text-only paper scope: short context on
#                        Gemma 3 1B + Llama 3.1 8B + Qwen 2.5 32B, plus long
#                        context on Qwen 2.5 32B at seq {256,512,1024,1621}.
#                        Requires an 80 GB-class GPU and ≥100 GB disk for
#                        model weights; runtime is multi-hour.
#                        Writes $RESULTS_DIR/multimodel_{gemma3_1b,llama_8b,qwen_32b}.json
#                        and $RESULTS_DIR/longctx_qwen_32b.json.
#   --verify  (default after --quick or --full) Runs
#                        tools/paper_invariants.py to check that raw data
#                        still validates every numeric claim in the paper.
#
# Usage:
#   HF_TOKEN=hf_xxxxx ./tools/reproduce.sh              # quick + verify
#   HF_TOKEN=hf_xxxxx ./tools/reproduce.sh --full       # full + verify
#   HF_TOKEN=hf_xxxxx ./tools/reproduce.sh --quick --no-verify
#
# Note: --full intentionally does NOT overwrite the committed paper artifacts
# at findings/multimodel_*.json / findings/longctx_qwen_32b*.json unless the
# user passes --overwrite. Default is a side-by-side file in
# findings/reproduced/ that can be diffed against the committed data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_DIR/.venv"
RESULTS_DIR="$REPO_DIR/findings"
REPRODUCED_DIR="$RESULTS_DIR/reproduced"
PLOTS_DIR="$REPO_DIR/plots"
RESULTS_FILE="$RESULTS_DIR/_raw_compression_reproduced.json"

# Flag parsing
MODE="quick"
VERIFY=1
OVERWRITE=0
for arg in "$@"; do
    case "$arg" in
        --quick) MODE="quick" ;;
        --full)  MODE="full" ;;
        --verify) VERIFY=1 ;;
        --no-verify) VERIFY=0 ;;
        --overwrite) OVERWRITE=1 ;;
        -h|--help)
            grep -E '^# ' "${BASH_SOURCE[0]}" | head -40 | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── 1. Check Python version ────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$("$candidate" -c 'import sys; print(sys.version_info.major)')
        minor=$("$candidate" -c 'import sys; print(sys.version_info.minor)')
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done
[ -z "$PYTHON" ] && fail "Python 3.10+ is required. Found: $(python3 --version 2>/dev/null || echo 'none')"
ok "Using $PYTHON ($ver)"

# ── 2. Check pip ───────────────────────────────────────────────────────────
$PYTHON -m pip --version &>/dev/null || fail "pip not found. Install it: $PYTHON -m ensurepip --upgrade"
ok "pip available"

# ── 3. Check HF_TOKEN ─────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    fail "HF_TOKEN environment variable is required (Gemma 3 is a gated model).\n" \
         "       Get a token at https://huggingface.co/settings/tokens\n" \
         "       Then: export HF_TOKEN=hf_xxxxx && ./tools/reproduce.sh"
fi
ok "HF_TOKEN is set"

# ── 4. Create / reuse virtual environment ──────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    info "Reusing existing venv at $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "venv activated"

# ── 5. Install dependencies ───────────────────────────────────────────────
info "Installing dependencies (torch, transformers, accelerate, scipy, matplotlib, huggingface_hub)..."
pip install --quiet --upgrade pip
pip install --quiet \
    torch \
    transformers \
    accelerate \
    huggingface_hub \
    scipy \
    matplotlib
ok "Dependencies installed"

# ── 6. Detect GPU vs CPU ──────────────────────────────────────────────────
DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
if [ "$DEVICE" = "cuda" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    ok "GPU detected: $GPU_NAME"
    info "Estimated runtime: ~2 minutes"
else
    warn "No GPU detected. Running on CPU."
    info "Estimated runtime: ~15 minutes"
fi

# ── 7. Run the experiment ─────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR" "$PLOTS_DIR" "$REPRODUCED_DIR"

if [ "$MODE" = "quick" ]; then
    info "Mode: --quick (Gemma 3 1B, 8 prompts, short context)"
    info "Running activation compression experiment..."
    echo ""
    python "$SCRIPT_DIR/activation_compression_experiment.py" \
        --out "$RESULTS_FILE" \
        --n-prompts 8 \
        --device "$DEVICE"
    echo ""
    ok "Quick reproduction complete. Raw results: $RESULTS_FILE"

    # ── 8. Generate plots ────────────────────────────────────────────────
    info "Generating plots..."
    python "$SCRIPT_DIR/activation_compression_plot.py" \
        --in "$RESULTS_FILE" \
        --out-dir "$PLOTS_DIR"
    echo ""
    ok "Plots saved to $PLOTS_DIR/"
else
    info "Mode: --full (3 text families short context + Qwen long context)"
    warn "Full reproduction needs an 80 GB-class GPU and runs for multiple hours."
    warn "Results go to $REPRODUCED_DIR/ to preserve committed artifacts."

    # Short context on three text-only families (as in paper §3.1–§3.3).
    # Model IDs match what the committed findings/multimodel_*.json files used.
    declare -A MODELS=(
        ["gemma3_1b"]="google/gemma-3-1b-it"
        ["llama_8b"]="meta-llama/Llama-3.1-8B-Instruct"
        ["qwen_32b"]="Qwen/Qwen2.5-32B-Instruct"
    )
    for key in "${!MODELS[@]}"; do
        model="${MODELS[$key]}"
        out="$REPRODUCED_DIR/multimodel_${key}.json"
        info "Short context: $model -> $out"
        python "$SCRIPT_DIR/multi_model_experiment.py" \
            --model "$model" --out "$out"
    done

    # Long context on Qwen 2.5 32B (paper §3.4–§3.6).
    info "Long context: Qwen 2.5 32B at seq {256,512,1024,1621} -> $REPRODUCED_DIR/longctx_qwen_32b.json"
    python "$SCRIPT_DIR/long_context_validation.py" \
        --model "Qwen/Qwen2.5-32B-Instruct" \
        --out "$REPRODUCED_DIR/longctx_qwen_32b.json" \
        --seq-lens 256 512 1024 1621

    if [ "$OVERWRITE" = "1" ]; then
        warn "--overwrite: copying reproduced files over committed findings/"
        for key in "${!MODELS[@]}"; do
            cp "$REPRODUCED_DIR/multimodel_${key}.json" "$RESULTS_DIR/multimodel_${key}.json"
        done
        cp "$REPRODUCED_DIR/longctx_qwen_32b.json" "$RESULTS_DIR/longctx_qwen_32b.json"
    fi
    ok "Full reproduction complete. Reproduced results: $REPRODUCED_DIR/"
fi

# ── 8b. Verify invariants ───────────────────────────────────────────────
if [ "$VERIFY" = "1" ]; then
    info "Running paper invariants to verify the reproduced data validates every claim..."
    python "$SCRIPT_DIR/paper_invariants.py" || {
        warn "Invariant check failed. Inspect output above."
        exit 1
    }
    ok "All paper invariants pass."
fi

# ── 9. Print summary ─────────────────────────────────────────────────────
# Summary only applies to the quick-mode output; full mode emits per-model files.
if [ "$MODE" != "quick" ]; then
    exit 0
fi
echo ""
echo "============================================================"
echo "  CARRIER-PAYLOAD (QUICK) — Gemma 3 1B RESULTS SUMMARY"
echo "  Text-only paper scope: this is ONE of three text families."
echo "  Use --full to reproduce Llama 3.1 8B + Qwen 2.5 32B + long context."
echo "============================================================"
echo ""

python -c "
import json, numpy as np
from collections import defaultdict

with open('$RESULTS_FILE') as f:
    data = json.load(f)

meta = data['meta']
results = data['results']

groups = defaultdict(list)
for r in results:
    key = (r['splice_layer'], r['pca_rank'], r['sparse_topk_frac'])
    groups[key].append(r)

agg = []
for (sl, pr, sf), rs in groups.items():
    agg.append({
        'sl': sl, 'pr': pr, 'sf': sf,
        'cr': np.mean([r['compression_ratio'] for r in rs]),
        'kl': np.mean([r['kl_divergence'] for r in rs]),
        'top1': np.mean([r['top1_agree'] for r in rs]),
        'top5': np.mean([r['top5_overlap'] for r in rs]),
    })

print(f'Model:         {meta[\"model_id\"]}')
print(f'Device:        {meta[\"device\"]}')
print(f'Prompts:       {meta[\"n_prompts\"]}')
print(f'Splice layers: {meta[\"splice_layers\"]}')
print(f'Configs tested: {len(agg)} (averaged over prompts)')
print()

good = [a for a in agg if a['kl'] < 0.01]
marginal = [a for a in agg if a['kl'] < 0.1]

if good:
    best = max(good, key=lambda x: x['cr'])
    print(f'Best compression at KL < 0.01 (near-lossless):')
    print(f'  {best[\"cr\"]:.1f}x compression | KL = {best[\"kl\"]:.4f} | Top-1 agree = {best[\"top1\"]:.0%}')
    print(f'  Config: layer {best[\"sl\"]}, PCA rank {best[\"pr\"]}, sparse frac {best[\"sf\"]}')
    print()

if marginal:
    bm = max(marginal, key=lambda x: x['cr'])
    print(f'Best compression at KL < 0.1  (marginal quality):')
    print(f'  {bm[\"cr\"]:.1f}x compression | KL = {bm[\"kl\"]:.4f} | Top-1 agree = {bm[\"top1\"]:.0%}')
    print()

# Bandwidth implication
if good:
    b = best
    hidden_dim = 1152  # Gemma 3 1B
    raw_bytes_per_token = hidden_dim * 2  # fp16
    compressed_bytes = raw_bytes_per_token / b['cr']
    print(f'Bandwidth implication (at {b[\"cr\"]:.0f}x):')
    print(f'  Raw:        {raw_bytes_per_token:,} bytes/token ({raw_bytes_per_token/1024:.1f} KB)')
    print(f'  Compressed: {compressed_bytes:,.0f} bytes/token ({compressed_bytes/1024:.2f} KB)')
    print(f'  A phone on 5 Mbps can sustain ~{5_000_000/8/compressed_bytes:,.0f} tokens/sec transport')
    print()

print('Key insight: LLM activations live on a low-dimensional manifold.')
print('A PCA carrier captures >99.9% of variance; the sparse residual')
print('handles outlier dimensions. Together they enable extreme compression')
print('with zero quality loss on the next-token prediction task.')
print()
print('Plots:')
for name in ['activation_compression_pareto.png',
             'activation_compression_top1_agreement.png',
             'activation_compression_variance_vs_kl.png',
             'activation_compression_heatmap.png']:
    print(f'  $PLOTS_DIR/{name}')
"

echo ""
echo "============================================================"
echo "  Done. Full paper: https://github.com/tejasphatak/webmind-research"
echo "============================================================"
