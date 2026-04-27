# STS-B Pattern Analysis — 100 Random Pairs

## Current: Spearman 0.6084. MiniLM: 0.8203. Gap: 0.21.

## Categories (100 pairs)
- CORRECT (error < 0.15): 35 pairs → system handles these well
- PARTIAL (error 0.15-0.50): 62 pairs → the GAP
- MISS_HIGH (gold>0.7, ULI<0.3): 1 pair → severe misses
- FALSE_HIGH (gold<0.3, ULI>0.5): 2 pairs → false positives

## THE 62 PARTIAL ERRORS — Pattern Analysis

### Pattern 1: VERB PARAPHRASE (most common, ~25 pairs)
```
"A man is slicing open a fish" vs "A man is cutting up a fish"
  gold=0.88 uli=0.429 shared={fish, man} unique_a={slice,open} unique_b={cutting}
  
"The cat is licking a bottle" vs "A cat plays with a small bottle"
  gold=0.47 uli=0.25 shared={bottle, cat} unique_a={lick} unique_b={play, small}
```
**What happens**: The NOUNS match (fish, man, cat, bottle) → Jaccard catches some overlap. But the VERBS differ (slice≠cut, lick≠play) and our system scores them 0.00-0.03 because WordNet doesn't put them in the same synset.

**What a human does**: "slicing" IS "cutting" — they're SYNONYMS. A human instantly knows this. WordNet DOES have them as synonyms (cut.v.01 includes "slice") but our meaning set expansion doesn't reach it because of POS mixing.

**The fix**: For VERB features specifically, use WUP graph distance (0.3-0.6) instead of meaning set overlap (0.01-0.05). WUP walks the taxonomy and finds the shared hypernym.

### Pattern 2: JACCARD FLOOR FOR UNRELATED (2 false positives)
```
"You don't have to worry" vs "You don't have to do anything to season it"
  gold=0.0 uli=0.556 J=0.556 shared_all={you, do, not, have, to, it}
```
**What happens**: These share 6 function words → Jaccard=0.556 → MAX takes Jaccard → ULI=0.556. But gold=0.0 (completely different meanings).

**What a human does**: Recognizes that "worry" vs "season" are the KEY content words, and they have nothing in common. The function words are structural, not meaningful.

**The fix**: Jaccard should be DISCOUNTED when content words don't match at all. If content_sim < threshold AND jaccard is driven by function words only → reduce.

### Pattern 3: SCORE COMPRESSION IN MIDDLE RANGE (~20 pairs)
```
"Two men pushed carts through the woods" vs "Two men are pushing carts"
  gold=0.70 uli=0.50 shared={cart, men, push, two} unique_a={wood}
```
**What happens**: Most content words match (4/5) → should score 0.7+. But our scoring averages multiple feature sets, compressing toward the middle.

**What a human does**: Counts shared vs unique — 4 shared, 1 unique = ~80% overlap.

**The fix**: The all-words Jaccard (4/5 = 0.8) IS capturing this correctly. But content_sim is low because the meaning expansion dilutes the direct word matches. The MAX should take Jaccard here.

### Pattern 4: MODIFIER DIFFERENCES (~10 pairs)
```
"Someone is on a blanket" vs "The person is making a blanket"
  gold=0.28 uli=0.53 shared={blanket} unique_a={someone} unique_b={make, person}
```
**What happens**: "on a blanket" ≠ "making a blanket" — same OBJECT, different ACTION/RELATION. Our system sees "blanket" shared → gives credit. But the VERB changes the meaning completely.

**What a human does**: Recognizes that the ACTION determines whether the pair is similar, not just the OBJECT.

**The fix**: When the VERB is in unique (different verbs), the similarity should be LOWER than when verbs match. Currently verbs and nouns contribute equally.

## SUMMARY: 3 Fixes Needed

1. **Verb graph for paraphrases**: Use WUP for verb feature matching (steer↔drive=0.31 vs meaning set 0.03). Only for VERBS — noun WUP is too noisy.

2. **Jaccard content gate**: When Jaccard > content_sim AND content_sim ≈ 0 → the Jaccard is driven by function words only → discount. "You don't have to worry" ≈ "You don't have to season it" should NOT score 0.55.

3. **Verb importance weighting**: When verbs DIFFER ("on blanket" vs "making blanket"), the overall score should be pulled DOWN more than when nouns differ.
