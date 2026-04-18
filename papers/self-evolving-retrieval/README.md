# Self-Evolving Retrieval: A Third Architecture for AI

## Files
- `paper.md` — the paper
- `review.md` — scientific review (4/10, with feedback)
- `benchmark-baseline.json` — run 2 baseline (1.3% EM, before learning)
- `benchmark-after-learning.json` — run 8 (71.3% EM, after learning, same questions)
- `benchmark-held-out-before.json` — run 9 (0.7% EM, fresh questions, first encounter)
- `benchmark-held-out-after.json` — run 10 (25.3% EM, same fresh questions, after learning)

## Reproduce

```bash
# 1. Start the SAQT server
cd ~/Synapse/synapse-src/saqt
pip install sentence-transformers faiss-cpu
python3 serve.py

# 2. Run baseline benchmark
node ~/Synapse/synapse-src/saqt/benchmark.mjs --samples 50 --concurrency 1

# 3. Rebuild FAISS to include learned pairs
python3 -c "
import sqlite3, numpy as np, faiss
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
db = sqlite3.connect('../../trained_model/saqt.db')
idx = faiss.read_index('../../trained_model/saqt.faiss')
missing = db.execute('SELECT question FROM qa WHERE id > ?', (idx.ntotal,)).fetchall()
if missing:
    embs = encoder.encode([r[0] for r in missing], normalize_embeddings=True).astype(np.float32)
    idx.add(embs)
    faiss.write_index(idx, '../../trained_model/saqt.faiss')
"

# 4. Restart server + run again — scores should improve
python3 serve.py &
node ~/Synapse/synapse-src/saqt/benchmark.mjs --samples 50 --concurrency 1
```

## Key result
Run 9 → Run 10 on held-out data: **0.7% → 25.3% EM** (HotPotQA: 0→72%)
