# SAQT Latency & Stress Test Results

**Date:** 2026-04-18T16:37:38.833Z
**Endpoint:** http://localhost:3002/api/saqt/query

## Server Health

| Metric | Before | After |
|--------|--------|-------|
| Ready | true | true |
| Chunks | 305,628 | 305,633 |
| Mode | faiss | faiss |

## Latency Profile (20 queries)

| Category | Count | RT Mean | RT Median | RT p95 | Server Mean | Server p95 | Avg Answer Len |
|----------|-------|---------|-----------|--------|-------------|------------|----------------|
| short | 10 | 3724ms | 4086ms | 4836ms | 3697ms | 4817ms | 240 chars |
| medium | 5 | 5106ms | 4419ms | 8096ms | 5083ms | 8067ms | 122 chars |
| long | 5 | 5226ms | 4931ms | 7429ms | 5200ms | 7381ms | 420 chars |

## Concurrency Stress Test

| Concurrency | Succeeded | Failed | Wall Time | Mean RT | p95 RT |
|-------------|-----------|--------|-----------|---------|--------|
| 5 | 5 | 0 | 3378ms | 3015ms | 3377ms |
| 10 | 10 | 0 | 4703ms | 3783ms | 4691ms |
| 20 | 20 | 0 | 7407ms | 5675ms | 7401ms |
| 50 | 50 | 0 | 23847ms | 12775ms | 16873ms |

## Throughput (Sequential)

- **Queries:** 30
- **Succeeded:** 30
- **Duration:** 137s
- **Sustained QPS:** 0.22
- **Mean RT:** 4559ms
- **p95 RT:** 7564ms
- **p99 RT:** 7669ms

## Cloudflare Path Overhead (3 samples, post-stress)

| Metric | Value |
|--------|-------|
| Server time (mean) | 1987ms |
| Round-trip via Cloudflare (mean) | 2426ms |
| Cloudflare/nginx overhead | ~440ms |

## Analysis

**Breaking point:** No failures observed up to 50 concurrent queries (after ThreadingMixIn fix).

**Critical finding -- BrokenPipeError crash:** The original single-threaded `HTTPServer` crashed on BrokenPipeError when clients disconnected mid-response. This killed the entire server process. Fixed by adding `ThreadingMixIn` and try/except around `wfile.write()`.

**CPU contention:** During benchmarking, 4 background encoding processes (SentenceTransformer rebuilds) consumed 100% CPU each on a 4-vCPU VM (load average 17.67). This inflated server-side latency from ~2s (idle) to ~4.5s (contended). The FAISS search itself is fast; the bottleneck is the SentenceTransformer `encoder.encode()` call per query competing for CPU.

**Latency breakdown (idle server):** ~2s server-side (encoding query + FAISS search) + ~440ms Cloudflare/TLS/nginx overhead = ~2.4s total.

**Latency breakdown (contended):** ~4.5s server-side + ~30ms localhost overhead = ~4.5s total.

**Concurrency scaling:** Linear degradation. 5 concurrent = 3.4s wall; 50 concurrent = 23.8s wall. No failures at any level.

**Throughput:** 0.22 qps sequential under heavy CPU contention. Expected ~0.5 qps when idle.

**Server healthy after stress:** YES (305,628 -> 305,633 chunks, 5 new pairs learned during test)
