"""
Shared constants for the reasoning engine.

All configurable thresholds, word sets, and user-facing strings
live here — single source of truth, no duplication across modules.
"""

# --- User-facing strings ---
# These are defaults. A future config layer can override them.
ABSTAIN_MESSAGE = "I don't know."
ABSTAIN_OOV_MESSAGE = "I don't know — none of those words are in my vocabulary."

# --- Word classification ---
# Function/stop words: high frequency, carry grammar not content.
# Used for: filtering query words, detecting generic neurons,
# preventing convergence jumps to non-content words.
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "and",
    "but", "or", "nor", "not", "no", "so", "yet", "both",
    "it", "its", "this", "that", "these", "those",
    "who", "what", "which", "where", "when", "how", "why",
})

# Structural words: appear in templates as fixed text (not slots).
# Superset of function words — includes common verbs that form
# sentence structure rather than carrying unique content.
STRUCTURAL_WORDS = FUNCTION_WORDS | frozenset({
    "wrote", "discovered", "invented", "created", "founded",
    "born", "died", "lived", "made", "built", "designed",
})

# --- Generation thresholds ---
# Successor walk: confidence above this = grammar token (fast path)
GRAMMAR_CONFIDENCE_THRESHOLD = 0.8

# Convergence jump: max number of sentence-boundary crossings
MAX_CONVERGENCE_JUMPS = 2

# Query anchor: minimum weight of query vector in context blend
# (prevents generation from forgetting what was asked)
QUERY_ANCHOR_FLOOR = 0.4

# Paragraph: sentences scoring below this fraction of the best
# sentence's score are excluded (prevents noise sentences)
PARAGRAPH_RELEVANCE_FLOOR = 0.5
