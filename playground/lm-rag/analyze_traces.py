"""
Analyze reasoning traces and extract generalizable patterns.

Reads reasoning_traces.jsonl, normalizes pattern names across models,
clusters by reasoning strategy, and outputs the patterns we need to
code into the engine.
"""

import json
import sys
import re
from collections import Counter, defaultdict

TRACE_FILE = 'reasoning_traces.jsonl'


def load_traces():
    traces = []
    with open(TRACE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return traces


# ============================================================
# Normalize pattern names — both models use different names
# for the same underlying strategy
# ============================================================

PATTERN_MAP = {
    # Single-hop direct lookup
    'direct_lookup': 'DIRECT_LOOKUP',
    'direct_fact_retrieval': 'DIRECT_LOOKUP',
    'direct_retrieval': 'DIRECT_LOOKUP',
    'direct_fact_lookup': 'DIRECT_LOOKUP',
    'single_hop_factoid': 'DIRECT_LOOKUP',
    'single_hop_retrieval': 'DIRECT_LOOKUP',
    'direct fact retrieval': 'DIRECT_LOOKUP',
    'locality retrieval': 'DIRECT_LOOKUP',

    # Single-hop with verification (surprising or contested answers)
    'single_hop_factoid_with_nuance': 'LOOKUP_AND_VERIFY',
    'single_hop_factoid_with_verification': 'LOOKUP_AND_VERIFY',
    'verification_loop': 'LOOKUP_AND_VERIFY',
    'multi_source_consensus': 'LOOKUP_AND_VERIFY',
    'consensus vs. controversy synthesis': 'LOOKUP_AND_VERIFY',
    'fact verification': 'LOOKUP_AND_VERIFY',
    'misconception_correction': 'LOOKUP_AND_VERIFY',

    # Multi-hop entity chain
    'multi_hop_entity_chain': 'MULTI_HOP_CHAIN',
    'chained_entity_resolution': 'MULTI_HOP_CHAIN',
    'sequential entity resolution': 'MULTI_HOP_CHAIN',
    'entity_relationship_traversal': 'MULTI_HOP_CHAIN',
    'locate-then-attribute': 'MULTI_HOP_CHAIN',
    'agent-attribute mapping': 'MULTI_HOP_CHAIN',
    'invention-geography link': 'MULTI_HOP_CHAIN',
    'event-participant-attribute chain': 'MULTI_HOP_CHAIN',
    'cultural-geography link': 'MULTI_HOP_CHAIN',
    'geographic-economic mapping': 'MULTI_HOP_CHAIN',
    'social graph traversal': 'MULTI_HOP_CHAIN',
    'multi_hop_decomposition': 'MULTI_HOP_CHAIN',
    'sequential_hop': 'MULTI_HOP_CHAIN',
    'multi_hop_three_step': 'MULTI_HOP_CHAIN',
    'multi_hop_shortcut': 'MULTI_HOP_CHAIN',
    'multi_hop_with_calculation': 'MULTI_HOP_CHAIN',
    'multi_hop_with_temporal_uncertainty': 'MULTI_HOP_CHAIN',

    # Multi-hop temporal
    'multi_hop_temporal': 'MULTI_HOP_TEMPORAL',
    'temporal_event_intersection': 'MULTI_HOP_TEMPORAL',
    'temporal-entity link': 'MULTI_HOP_TEMPORAL',

    # Temporal (current state)
    'temporal_current_state': 'TEMPORAL_VERIFY',
    'temporal_validation': 'TEMPORAL_VERIFY',
    'temporal_most_recent': 'TEMPORAL_VERIFY',
    'temporal_self_referential': 'TEMPORAL_VERIFY',
    'temporal_with_verification': 'TEMPORAL_VERIFY',
    'active_incumbent_check': 'TEMPORAL_VERIFY',
    'chronological_scan': 'TEMPORAL_VERIFY',
    'existence_verification': 'TEMPORAL_VERIFY',
    'temporal_event_check': 'TEMPORAL_VERIFY',
    'leadership_stability_check': 'TEMPORAL_VERIFY',
    'temporal_product_release': 'TEMPORAL_VERIFY',
    'software_version_tracking': 'TEMPORAL_VERIFY',
    'product_lifecycle_check': 'TEMPORAL_VERIFY',

    # Comparison
    'direct attribute comparison': 'COMPARE',
    'temporal sequence analysis': 'COMPARE',
    'data point comparison': 'COMPARE',
    'physical property retrieval': 'COMPARE',
    'normalized data comparison': 'COMPARE',
    'index-based comparison': 'COMPARE',
    'parallel_lookup_compare': 'COMPARE',
    'attribute_comparison': 'COMPARE',
    'comparative_retrieval': 'COMPARE',
    'comparative_retrieval_with_insufficient_user_context': 'COMPARE',

    # Why/How (explanation)
    'multi_step_factual_enrichment': 'EXPLAIN',
    'multi_factor_synthesis': 'EXPLAIN',
    'theory_synthesis': 'EXPLAIN',
    'search_explain_verify': 'EXPLAIN',
    'biological process extraction': 'EXPLAIN',
    'technical mechanism analysis': 'EXPLAIN',
    'physics principle retrieval': 'EXPLAIN',
    'molecular process explanation': 'EXPLAIN',
    'systems architecture retrieval': 'EXPLAIN',
    'wave physics extraction': 'EXPLAIN',
    'computational process retrieval': 'EXPLAIN',
    'engineering mechanism retrieval': 'EXPLAIN',
    'economic system analysis': 'EXPLAIN',

    # Disambiguation
    'detect_ambiguity_present_all': 'DISAMBIGUATE',
    'disambiguation_retrieval': 'DISAMBIGUATE',

    # Negation
    'enumerate_then_exclude': 'ENUMERATE_EXCLUDE',

    # Unanswerable
    'recognize_unanswerable': 'UNANSWERABLE',

    # Link/content analysis
    'content_analysis': 'FETCH_AND_ANALYZE',

    # Product comparison
    'multi_attribute_comparison': 'MULTI_ATTRIBUTE_COMPARE',

    # Creative
    'generative': 'GENERATE',

    # Calculation
    'direct_calculation': 'CALCULATE',

    # Logic
    'logical_deduction': 'REASON',
    'multi_step_deduction': 'REASON',
}


def normalize_pattern(raw):
    """Normalize a pattern name to a canonical form."""
    key = raw.lower().strip().replace('-', '_').replace('→', '_')
    if key in PATTERN_MAP:
        return PATTERN_MAP[key]
    # Fuzzy match on known map
    for k, v in PATTERN_MAP.items():
        if k in key or key in k:
            return v

    # Keyword-based fallback for Opus's creative names
    if any(w in key for w in ['search', 'judge', 'extract', 'give_up']):
        # It's an action sequence used as a pattern name — classify by structure
        if 'give_up' in key:
            return 'UNANSWERABLE'
        if 'synthesize' in key or 'synthesis' in key:
            return 'EXPLAIN'
        if 'calculate' in key:
            return 'CALCULATE'
        return 'DIRECT_LOOKUP'
    if any(w in key for w in ['comparison', 'compare', 'scaling', 'trade_off',
                               'pro_con', 'bifurcation', 'ecosystem']):
        return 'COMPARE'
    if any(w in key for w in ['fetch', 'url', 'content']):
        return 'FETCH_AND_ANALYZE'
    if any(w in key for w in ['procedural', 'instruction', 'step_by_step',
                               'checklist', 'planning', 'itinerary', 'optimization',
                               'framework', 'simplified']):
        return 'ADVICE_PROCEDURAL'
    if any(w in key for w in ['subjective', 'opinion', 'perspective', 'consensus',
                               'tension', 'divergent', 'debate']):
        return 'OPINION_SYNTHESIS'
    if any(w in key for w in ['disambigu', 'ambiguous', 'multi_sense', 'polysemous',
                               'named_entity', 'entity_vs_common']):
        return 'DISAMBIGUATE'
    if any(w in key for w in ['negation', 'set_subtraction', 'complementary',
                               'exception', 'exclusion']):
        return 'ENUMERATE_EXCLUDE'
    if any(w in key for w in ['unanswerable', 'predictive_failure', 'boundary_recognition',
                               'speculation']):
        return 'UNANSWERABLE'
    if any(w in key for w in ['math_', 'percentage', 'unit_conversion', 'arithmetic',
                               'exponent', 'discount', 'algebra', 'rate', 'sales_tax']):
        return 'CALCULATE'
    if any(w in key for w in ['locate_then', 'agent_attribute', 'invention_geography',
                               'event_participant', 'temporal_entity', 'geographic_economic',
                               'cultural_geography', 'multi_entity', 'entity_recognition',
                               'template_filling', 'clustered']):
        return 'MULTI_HOP_CHAIN'
    if any(w in key for w in ['creative', 'generation', 'construction', 'style',
                               'causal_chain']):
        return 'GENERATE'
    if any(w in key for w in ['financial', 'investment', 'risk', 'asset',
                               'incentive', 'constraint', 'trend', 'extrapolation']):
        return 'MULTI_FACTOR_ANALYSIS'
    if any(w in key for w in ['myth', 'busting', 'nuance', 'discovery']):
        return 'LOOKUP_AND_VERIFY'
    if any(w in key for w in ['multi_method', 'multi_source', 'aggregation',
                               'option', 'domain', 'categorization', 'evaluation',
                               'criteria', 'persona', 'biological', 'technical',
                               'recommendation', 'filtering']):
        return 'MULTI_FACTOR_ANALYSIS'

    return 'OTHER'


def extract_action_sequence(trace):
    """Extract the sequence of actions as a tuple."""
    return tuple(s.get('action', '?') for s in trace.get('steps', []))


def analyze(traces):
    print(f"Total traces: {len(traces)}")
    print(f"Models: {Counter(t.get('model', '?') for t in traces)}")
    print()

    # Normalize patterns
    for t in traces:
        t['normalized_pattern'] = normalize_pattern(t.get('pattern', 'unknown'))

    # Pattern distribution
    patterns = Counter(t['normalized_pattern'] for t in traces)
    print("=" * 60)
    print("NORMALIZED REASONING PATTERNS")
    print("=" * 60)
    for p, count in patterns.most_common():
        pct = count * 100 // len(traces)
        print(f"  {p:35s} {count:4d} ({pct:2d}%)")

    # Action sequence analysis
    print(f"\n{'=' * 60}")
    print("ACTION SEQUENCES (most common)")
    print("=" * 60)
    sequences = Counter(extract_action_sequence(t) for t in traces)
    for seq, count in sequences.most_common(15):
        print(f"  {' → '.join(seq):60s} {count:3d}")

    # Category × Pattern matrix
    print(f"\n{'=' * 60}")
    print("CATEGORY → PATTERN MAPPING")
    print("=" * 60)
    cat_pattern = defaultdict(Counter)
    for t in traces:
        cat = t.get('category', 'unknown')
        cat_pattern[cat][t['normalized_pattern']] += 1

    for cat, pats in sorted(cat_pattern.items()):
        top = pats.most_common(3)
        top_str = ', '.join(f"{p}({c})" for p, c in top)
        print(f"  {cat:25s} → {top_str}")

    # Convergence analysis
    print(f"\n{'=' * 60}")
    print("CONVERGENCE")
    print("=" * 60)
    converged = sum(1 for t in traces if t.get('converged'))
    print(f"  Converged: {converged}/{len(traces)} ({converged*100//len(traces)}%)")

    # By pattern
    for p, count in patterns.most_common():
        p_traces = [t for t in traces if t['normalized_pattern'] == p]
        conv = sum(1 for t in p_traces if t.get('converged'))
        avg_searches = sum(t.get('total_searches', 0) for t in p_traces) / len(p_traces)
        avg_steps = sum(len(t.get('steps', [])) for t in p_traces) / len(p_traces)
        print(f"  {p:35s} conv={conv}/{len(p_traces)} "
              f"avg_searches={avg_searches:.1f} avg_steps={avg_steps:.1f}")

    # Confidence distribution
    print(f"\n{'=' * 60}")
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 60)
    all_confs = []
    for t in traces:
        for s in t.get('steps', []):
            c = s.get('confidence', 0)
            if isinstance(c, (int, float)):
                all_confs.append(c)

    if all_confs:
        bins = [0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]
        labels = ['0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.85', '0.85-0.95', '0.95-1.0']
        for i in range(len(bins)-1):
            count = sum(1 for c in all_confs if bins[i] <= c < bins[i+1])
            bar = '█' * (count * 40 // len(all_confs))
            print(f"  {labels[i]:10s} {count:4d} {bar}")

    # Key insight: what ACTIONS follow what
    print(f"\n{'=' * 60}")
    print("ACTION TRANSITIONS (what follows what)")
    print("=" * 60)
    transitions = Counter()
    for t in traces:
        steps = t.get('steps', [])
        for i in range(len(steps) - 1):
            a1 = steps[i].get('action', '?')
            a2 = steps[i+1].get('action', '?')
            transitions[(a1, a2)] += 1

    for (a1, a2), count in transitions.most_common(20):
        print(f"  {a1:15s} → {a2:15s} {count:4d}")

    # The money shot: generalized patterns for the engine
    print(f"\n{'=' * 60}")
    print("GENERALIZED PATTERNS FOR ENGINE IMPLEMENTATION")
    print("=" * 60)
    print("""
Based on trace analysis, the engine needs these reasoning strategies:

1. DIRECT_LOOKUP
   Trigger: Simple factual question (who/what/when/where)
   Flow: SEARCH → JUDGE → EXTRACT
   Searches: 1 | Confidence threshold: >0.85

2. LOOKUP_AND_VERIFY
   Trigger: Answer seems surprising, contested, or counter-intuitive
   Flow: SEARCH → JUDGE(uncertain) → SEARCH(verify) → JUDGE → EXTRACT
   Searches: 2 | When to verify: confidence < 0.85 on first judge

3. MULTI_HOP_CHAIN
   Trigger: Question requires chaining facts (entity → attribute → entity)
   Flow: DECOMPOSE → [SEARCH → JUDGE → EXTRACT]×N → SYNTHESIZE
   Searches: 2-4 | Key: decompose into sub-questions FIRST

4. TEMPORAL_VERIFY
   Trigger: Question about current/recent/latest state
   Flow: SEARCH → JUDGE(freshness) → SEARCH(verify current) → EXTRACT
   Searches: 2 | Key: always cross-reference temporal answers

5. COMPARE
   Trigger: "which is better/larger/faster", comparison between entities
   Flow: DECOMPOSE → [SEARCH entity_A, SEARCH entity_B] → SYNTHESIZE
   Searches: 2-3 | Key: parallel lookup, then compare values

6. EXPLAIN
   Trigger: Why/how questions requiring mechanism explanation
   Flow: SEARCH → JUDGE(completeness) → [SEARCH more if incomplete] → SYNTHESIZE
   Searches: 1-2 | Key: judge whether explanation covers the causal chain

7. DISAMBIGUATE
   Trigger: Search returns multiple unrelated meanings
   Flow: SEARCH → JUDGE(ambiguous) → present all interpretations or ask
   Searches: 1-2 | Key: detect ambiguity in search results

8. ENUMERATE_EXCLUDE
   Trigger: Negation questions ("which does NOT")
   Flow: SEARCH(full set) → enumerate → exclude → EXTRACT
   Searches: 1-2 | Key: find the complete set first

9. FETCH_AND_ANALYZE
   Trigger: User provides a URL/link to analyze
   Flow: FETCH(url) → JUDGE(content type) → EXTRACT/SUMMARIZE
   Searches: 1 + fetch | Key: must access the content

10. CALCULATE
    Trigger: Mathematical expression or word problem
    Flow: CALCULATE(expression) or DECOMPOSE → CALCULATE → EXTRACT
    Searches: 0 | Key: no search needed, use math tool

11. GENERATE
    Trigger: Creative task (write, suggest, plan, design)
    Flow: [optional SEARCH for reference] → GENERATE
    Searches: 0-1 | Key: instruct model generates, not retrieves

12. UNANSWERABLE
    Trigger: Search yields nothing factual, question is philosophical/predictive
    Flow: SEARCH → JUDGE(no factual answer) → GIVE_UP(explain why)
    Searches: 1 | Key: detect early, don't keep searching
""")


if __name__ == '__main__':
    traces = load_traces()
    analyze(traces)
