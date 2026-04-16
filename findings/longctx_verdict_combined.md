# Long-Context Validation Verdict

Generated from: findings/longctx_qwen_32b.json, findings/longctx_gemma4_31b.json

```
======================================================================
LONG-CONTEXT VALIDATION VERDICT
======================================================================

### Qwen/Qwen2.5-32B-Instruct
   seq_len     rank 99%      bound   rank/bound
       256          8.4        256         3.3%
       512         55.0        512        10.7%
      1024        193.6       1024        18.9%
      1621        384.0       1621        23.7%

  log-log slope = 2.06
  ✗ VERDICT: rank scales linearly-ish with seq_len. P1 low-rank claim DOES NOT HOLD universally.

### google/gemma-4-31B-it
   seq_len     rank 99%      bound   rank/bound
       256        214.3        256        83.7%
       512        401.3        512        78.4%
      1024        710.0       1024        69.3%

  log-log slope = 0.86
  ✗ VERDICT: rank scales linearly-ish with seq_len. P1 low-rank claim DOES NOT HOLD universally.
```
