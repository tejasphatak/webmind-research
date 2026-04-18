# Papers

Each paper has its own directory with everything needed to reproduce it.

| Paper | Directory | Status | Key Result |
|-------|-----------|--------|------------|
| **Self-Evolving Retrieval** | [self-evolving-retrieval/](self-evolving-retrieval/) | Benchmarked | 0.7% → 25.3% EM on held-out data |
| Activation Speculation Dead | [activation-speculation-dead/](activation-speculation-dead/) | Published | Negative result — bootstrap deadlock |
| MoeMoe Resilience | [moemoe/](moemoe/) | Preliminary | Node failure recovery |
| SAQT Distributed Cognition | [saqt-distributed-cognition/](saqt-distributed-cognition/) | Draft | Distributed knowledge mesh |
| SAQT Ethics | [saqt-ethics/](saqt-ethics/) | Draft | Safety through data |
| SFCA Credit Assignment | [sfca/](sfca/) | Pre-registered | Shapley-fair attribution |
| Synapse v2 | [synapse-v2/](synapse-v2/) | Draft | Distributed specialists |

## Discipline
- Every paper lives with its data, code, and results
- Null results published — we don't file-drawer
- All code released MIT, papers CC-BY 4.0

## Running experiments

```bash
pip install -r ../requirements.txt
```

> **Note:** Some experiments depend on large model files or specific hardware. If you can't reproduce a result, please [open an issue](https://github.com/tejasphatak/webmind-research/issues).
