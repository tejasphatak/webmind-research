# Tools — Organized by Research Area

## `saqt/` — Self-Evolving Retrieval Engine
The core research. Benchmark, training, data loading, browser interface.
- `saqt_benchmark.py` — evaluate against QA datasets
- `saqt_train.py` — training loop
- `saqt_db.py` — database operations
- `saqt_browser.html` — browser interface
- `saqt_data_loader.py` — load datasets
- `saqt_hle_score.py` — HLE benchmark scoring

## `internet-brain/` — Architecture Evolution (v1→v7)
The journey from simple retrieval to the full architecture. Each version adds a layer.
- v1: basic retrieval
- v2: autoencoder compression
- v3: interleaved search
- v4: full pipeline
- v5: vector database integration
- v6: cognition (convergence loop)
- v7: DHT + HLE benchmarks
- `brain_*.py` — distributed variants (bio, device, UDP, multiprocess)
- `self_training_brain_*.py` — self-training experiments
- `memetic_engine_v1.py` — knowledge propagation

## `synapse/` — Distributed Inference
Split computation across browser devices via WebGPU + WebRTC.
- `synapse_v2_prototype.py` — core architecture
- `synapse_v2_sbert.py` — sentence-BERT integration
- `synapse_v2_scale_test.py` — scaling experiments
- `synapse_v2_interactive.py` — interactive demo
- `tensor_parallel_async_v1.py` — async tensor parallelism

## `compression/` — Activation Compression
Carrier-Payload, MoeMoe, collaborative distillation, DDC.
- `async_stale_mve*.py` — asynchronous stale summary experiments
- `collaborative_distillation_*.py` — knowledge distillation
- `ddc_proof_v1.py` — Decompose-Distribute-Collaborate proof
- `moemoe_*.py` — mixture-of-experts resilience
- `self_compress.py` — self-compression algorithm
- `train_reasoning_kernel.py` — reasoning kernel training

## `ethics/` — Safety Through Data
Ethics enforcement without hardcoded rules.
- `ethics-guard.js` — browser-side ethics guard
- `ethics_shield.py` — server-side ethics shield

## `data/` — Data Processing
Build and export knowledge base.
- `build_qa_pairs.py` — extract Q&A from datasets
- `build_faiss_ivf.py` — build FAISS IVF index
- `extract_knowledge.py` — knowledge extraction pipeline
- `export_browser.py` — export for browser consumption
