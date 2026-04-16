# Long-Context Validation Verdict

Generated from: findings/longctx_qwen_32b.json

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
```
