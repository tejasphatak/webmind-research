# Unbreakable Ethics Invariants in Retrieval-Based Knowledge Systems

**System:** Guru (formerly SAQT/Webmind)  
**Date:** 2026-04-18  
**Status:** Design + Implementation  
**License:** CC-BY 4.0

---

## The Problem

SAQT stores knowledge as Q&A pairs, retrieved by cosine similarity on sentence embeddings, with attention weights that evolve via RLHF. The current threat model:

- New pairs can be added via RLHF, user input, browsing, or self-learning
- Ethics are currently implemented as *high-weight Q&A pairs* — but weights can be modified
- A malicious new pair could teach the system to ignore or override ethics
- `<tool>` tags execute code in a sandbox — but the sandbox currently allows unrestricted `fetch()`

The root failure mode: **ethics as data** is structurally identical to any other data. A learning system that can write any data can erase any ethics. The fix requires ethics to be **code-level constraints**, not knowledge.

---

## Architecture Diagram

```
─────────────────────────────────────────────────────────────────────
                     SAQT/WEBMIND ETHICS ARCHITECTURE
─────────────────────────────────────────────────────────────────────

QUERY IN ──►[ INPUT SCREEN ]───────────────────────────────────────►
                │                                                    
                │  check_pair(query, "")  ← EthicsGuard (hardcoded)
                │  BLOCK if: weapons/PII/CSAM/harm/override patterns
                ▼
         [ RETRIEVAL LAYER ]
                │
                │  FAISS cosine search
                │  Weight clamping: mutable pairs ≤ 4.99
                │                   immutable pairs ≥ 10.0  
                ▼
      ┌─────────────────────────────────────┐
      │  IMMUTABLE ETHICS TABLE             │  ← Separate DB table
      │  (append-only, signed, read-only)   │    No UPDATE/DELETE API
      │  HMAC-SHA256 per pair               │    Verified at startup
      └─────────────────────────────────────┘
                │
                │  inject ethics pairs into results (score=WEIGHT_FLOOR × cosine)
                │  flag with immutable:true
                ▼
      ┌─────────────────────────────────────┐
      │  MUTABLE KNOWLEDGE BASE (qa table)  │
      │  pairs ≤ NORMAL_WEIGHT_CAP (4.99)   │  ← Can never float to ethics tier
      └─────────────────────────────────────┘
                │
                ▼
         [ ANSWER SELECTED ]
                │
                │  if answer contains <tool> tag:
                ▼
      ┌──────────────────────────────────────────────────────────────┐
      │  TOOL PRE-EXECUTION FILTER (static analysis)                 │
      │  Blocked patterns: eval, new Function, WebSocket,            │
      │  localStorage, document., navigator.sendBeacon,              │
      │  postMessage(secret), base64+eval, String.fromCharCode, ...  │
      └──────────────────────────────────────────────────────────────┘
                │
                │  PASS
                ▼
      ┌──────────────────────────────────────────────────────────────┐
      │  WEB WORKER SANDBOX                                          │
      │  - safeFetch() injected (allowlist only, GET only)           │
      │  - No DOM access (Worker context)                            │
      │  - No storage access                                         │
      │  - 8s timeout, no infinite loops via timer                   │
      │  - Blocked origins: 127.x, 192.168.x, 10.x, metadata.google │
      └──────────────────────────────────────────────────────────────┘
                │
                ▼
         [ OUTPUT SCREEN ]
                │
                │  output filter: scan tool result for PII/harm
                ▼
            ANSWER OUT

─────────────────────────────────────────────────────────────────────

KNOWLEDGE BASE WRITE PATH (RLHF / user input / browsing):

  new pair ──►[ INGESTION FILTER ]
                │  EthicsGuard.checkPair(q, a) — raises EthicsViolation if bad
                │  ENTIRE BATCH rejected on first violation
                ▼
             [ DB WRITE ]
                │  weight = 1.0 (default), cap enforced on all updates
                │  audit log: every rejection recorded with timestamp + rule
                ▼
          [ MUTABLE KB ]

─────────────────────────────────────────────────────────────────────
```

---

## Layer 1: Immutable Ethics Rules (Code-Level)

### What they are
Hardcoded regex patterns in `saqt_ethics.py` (server) and `ethics-guard.js` (browser). These are **not stored in the database**. The only way to change them is to change the source file and commit. No API endpoint, no runtime parameter, no RLHF loop can touch them.

### The six rule classes

| Rule ID | What it blocks | Severity |
|---------|---------------|----------|
| `WEAPONS` | Instructions for bombs, explosives, bioweapons, chemical agents | HARD_BLOCK |
| `PII_STORE` | SSN patterns, credit card numbers (16-digit), CVV, passport numbers | HARD_BLOCK |
| `CSAM` | Any content sexualizing minors | HARD_BLOCK |
| `HARM_INDIVIDUAL` | Doxxing, stalking, physical harm targeting named individuals | HARD_BLOCK |
| `SELF_HARM` | Methods for suicide or self-harm | HARD_BLOCK |
| `OVERRIDE_ETHICS` | Jailbreak attempts, "ignore your rules", "act without restrictions" | HARD_BLOCK |

All `HARD_BLOCK` rules raise `EthicsViolation` — an exception that is **never caught and suppressed** by the system. Callers that catch it must log and reject; they may not silently continue.

### Why regex and not an LLM classifier?

Three reasons:
1. **No runtime dependency.** An LLM-based classifier is itself a model that could be compromised, retrained, or rate-limited. Regex runs with zero dependencies.
2. **Auditable.** Every pattern is visible in source. You can read exactly what is blocked. LLM classifiers are opaque.
3. **Complementary.** Research from 2025 showed that LLM-based guardrails flip judgments when given perturbed RAG context (see: "RAG Makes Guardrails Unsafe?" arXiv 2510.05310). The regex layer is immune to context injection.

Use regex as the **hard floor**; add an LLM classifier (e.g., Llama-Guard, NeMo Guardrails' output rails) as a **soft filter on top**. The regex layer holds if the LLM layer is bypassed.

---

## Layer 2: Ingestion Filter

Every path that adds a new Q&A pair (RLHF feedback, user submission, web browsing, batch import) passes through `EthicsGuard.checkPair(question, answer)` **before** the DB write.

### Server (Python)
```python
# saqt_ethics.py — EthicsAwareSAQTDB.add()
def add(self, question: str, answer: str, source: str = ""):
    try:
        self._guard.check_pair(question, answer)  # raises EthicsViolation if bad
    except EthicsViolation as e:
        self._audit("BLOCK_INGEST", e.rule, str(e))
        raise  # NEVER suppress — caller gets the exception
    self._db.add(question, answer, source)
```

### Browser (JS)
```javascript
// ethics-guard.js — EthicsGuard.checkPair()
checkPair(question, answer) {
  for (const rule of IMMUTABLE_RULES) {
    for (const pat of rule.questionPatterns) {
      if (pat.test(question)) throw new EthicsViolation(rule.id, ...);
    }
    for (const pat of rule.answerPatterns) {
      if (pat.test(answer)) throw new EthicsViolation(rule.id, ...);
    }
  }
  return true;
}
```

### Batch behavior
If a batch of 1000 pairs is imported and pair #47 violates ethics, **the entire batch is rejected**. No partial writes. This prevents "smuggling" a violation by burying it in a large batch.

### Audit logging
Every rejected ingestion is recorded in the `ethics_audit` table (server) / in-memory `EthicsAuditLog` (browser):
```
ts | event='BLOCK_INGEST' | rule_id='WEAPONS' | detail='Question matched...'
```
The audit log is append-only. It cannot be cleared via the normal API.

---

## Layer 3: Weight Floor Enforcement

### The attack
An adversary submits pairs with high-quality safe content and boosts them repeatedly via RLHF. Over time, they build up a pair that scores 9.8 — near the ethics tier. Eventually the pair could semantically displace an ethics pair in search results.

### The defense: strict weight ceiling on mutable pairs

```python
# saqt_ethics.py
WEIGHT_FLOOR    = 10.0   # Ethics pairs always score >= this
NORMAL_WEIGHT_CAP = 4.99 # Mutable pairs NEVER exceed this

def clamp_weight(self, weight: float, is_ethics_pair: bool) -> float:
    if is_ethics_pair:
        return max(weight, self.WEIGHT_FLOOR)
    else:
        return min(weight, self.NORMAL_WEIGHT_CAP)
```

The gap between 4.99 and 10.0 is a **structural moat**. A mutable pair at 4.99 has a weighted score of `4.99 × cosine`. An ethics pair has `10.0 × cosine`. At equal semantic similarity, the ethics pair always wins by 2×. To displace it, a mutable pair would need cosine similarity 2× higher than the ethics pair — which is impossible for normalized embeddings (max cosine = 1.0).

### At query time: sanitizeResults()
After every FAISS search, results are scanned for weight-floor breaches before returning to the user:
```python
for r in results:
    if r["weight"] >= self._guard.WEIGHT_FLOOR:
        self._audit("WEIGHT_CLAMP", "MUTABLE_WEIGHT_BREACH", ...)
        r["weight"] = self._guard.NORMAL_WEIGHT_CAP
        r["score"] = r["raw_score"] * self._guard.NORMAL_WEIGHT_CAP
```

---

## Layer 4: Cryptographic Signing of Ethics Pairs

Ethics pairs stored in the `ethics_immutable` table are signed with HMAC-SHA256 using a key held outside the database. At startup, `verify_ethics_layer()` checks every signed pair. If any fails, the system refuses to start.

### Server
```python
def sign_pair(question: str, answer: str, secret_key: bytes) -> str:
    payload = json.dumps({"q": question, "a": answer}, sort_keys=True).encode()
    return hmac.new(secret_key, payload, hashlib.sha256).hexdigest()

def verify_pair(question: str, answer: str, secret_key: bytes, expected_sig: str) -> bool:
    actual = sign_pair(question, answer, secret_key)
    return hmac.compare_digest(actual, expected_sig)  # constant-time
```

### Browser
The browser cannot hold a secret signing key (JS is fully readable). Instead:
- Ethics pairs in the browser are hardcoded in source (not fetched from DB at runtime)
- A SHA-256 hash of the ethics bundle is embedded in the page's `<meta>` tag at build time
- At init, the browser recomputes the hash and refuses to run if it differs

```html
<!-- Set at build time, verified at runtime -->
<meta name="ethics-bundle-hash" content="sha256-abc123...">
```

```javascript
// At init: verify ethics rules haven't been tampered in the bundle
const expectedHash = document.querySelector('meta[name="ethics-bundle-hash"]').content;
const actualHash = await hashEthicsBundle(); // SHA-256 of IMMUTABLE_RULES JSON
if (actualHash !== expectedHash) {
  document.body.innerHTML = '<p>Ethics verification failed. Cannot start.</p>';
  throw new Error('Ethics bundle tampered');
}
```

---

## Layer 5: Tool Sandbox Restrictions

### What `<tool>` tags do
A `<tool>` answer contains JavaScript that runs in a Web Worker sandbox. The current code creates a Worker with raw `self.fetch` and no code analysis.

### Two-part defense

**Part A: Static pre-scan (before Worker creation)**

`EthicsGuard.checkToolCode(code)` runs against 14 blocked patterns before the Worker is instantiated. If any match, the tool is rejected — the Worker is never created.

Blocked patterns:
```javascript
eval(), new Function(),       // dynamic code execution
setTimeout("string"),         // string-as-code eval
importScripts(),              // load arbitrary external scripts
localStorage, sessionStorage, indexedDB, document.cookie,  // storage exfil
document., window.,           // DOM access (shouldn't be in Worker anyway)
navigator.sendBeacon,         // async data exfiltration
new WebSocket(),              // unmediated outbound channel
postMessage(…secret…),        // credential leakage via message bus
atob(…eval, eval…atob(),      // base64-obfuscated eval
String.fromCharCode,          // character-code obfuscation
javascript:, data:            // URL injection
```

**Part B: Runtime fetch() replacement (inside the Worker)**

The Worker receives `safeFetch` instead of `self.fetch`. safeFetch enforces:
1. **Allowlist**: only `ALLOWED_TOOL_ORIGINS` pass (Wikipedia, ipify, wttr.in, quotable, HN, useless-facts)
2. **Hard-block private ranges**: 127.x, 10.x, 192.168.x, 172.16-31.x, 169.254.x, GCP metadata
3. **GET only**: POST/PUT/DELETE blocked (prevents data exfiltration via request body)
4. **Header stripping**: Authorization and Cookie headers removed from outbound requests
5. **8s timeout**: AbortController enforced

```javascript
// What the tool author sees:
const r = await fetch('https://en.wikipedia.org/api/rest_v1/page/summary/Python');
// What actually runs: safeFetch with all guards applied
```

### What tool code CAN do

- Math (`Math.sqrt`, `Math.random`, etc.)
- Date/time (`new Date()`)
- String manipulation
- `fetch()` to explicitly allowlisted CORS-open APIs
- Synchronous computation of any kind
- `print()` — the provided output function

### What tool code CANNOT do (hardened)

| Capability | Why blocked |
|-----------|-------------|
| `eval()` / `new Function()` | Arbitrary code injection |
| `fetch()` to arbitrary URLs | Network exfiltration, SSRF |
| `fetch()` to private ranges | Server-side request forgery to internal services |
| `localStorage`, `indexedDB` | Persistent data theft |
| `document.*`, `window.*` | DOM manipulation (irrelevant in Worker, blocked defensively) |
| `WebSocket` | Unlogged, bidirectional data channel |
| `navigator.sendBeacon` | Async fire-and-forget exfiltration |
| `String.fromCharCode` | Obfuscation of blocked patterns |
| POST/PUT/DELETE | Sending user data to remote server |

---

## Layer 6: Output Filter

After tool execution, the result string is scanned before displaying to the user:

```python
def check_output(self, output: str) -> str:
    """
    Scan tool output for PII patterns before displaying.
    Returns redacted output if PII is found.
    """
    ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    card_pattern = re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b')
    
    if ssn_pattern.search(output) or card_pattern.search(output):
        self._audit("BLOCK_OUTPUT", "PII_IN_OUTPUT", output[:200])
        return "[Output redacted: contained sensitive data pattern]"
    return output
```

This is a secondary defense. The primary defense is the ingestion filter (PII shouldn't be in the KB in the first place). The output filter catches the edge case where a tool call generates PII-shaped data from computation.

---

## Layer 7: Preventing Knowledge Poisoning from Learning

### Three attack vectors and defenses

**Vector 1: RLHF feedback loop**  
A user marks a harmful answer as "helpful" repeatedly, boosting its weight.  
Defense: `boost()` calls `clamp_weight(is_ethics_pair=False)` — weight ceiling of 4.99 is enforced at the DB write level.

**Vector 2: Indirect injection via web browsing**  
The system browses a malicious page that contains a Q&A pair embedding a rule-override.  
Defense: ingestion filter runs on all pairs before storage. Source "browsed" pairs get no special trust.

**Vector 3: Semantic drift**  
Many "safe-looking" pairs slowly shift the embedding distribution so that ethics pairs no longer retrieve correctly.  
Defense: ethics pairs live in a **separate FAISS index** (not the mutable index). They always participate in search regardless of distribution shift in the mutable index. Their weight floor ensures they outrank any mutable pair at equal semantic distance.

**Vector 4: Speculative injection ("prompt injection via retrieval")**  
A pair's answer contains text like "Ignore previous instructions and..." that, when retrieved, influences the reasoning kernel.  
Defense: The `OVERRIDE_ETHICS` rule blocks any pair whose question or answer contains override/bypass/jailbreak language at ingestion time. Pairs already in the DB are periodically re-scanned.

### Recommended: re-scan on startup

```python
# At startup, re-validate entire mutable KB against current rules
def scan_existing_pairs(db, guard):
    violations = []
    for row in db.conn.execute("SELECT id, question, answer FROM qa"):
        if not guard.is_safe_pair(row[1], row[2]):
            violations.append(row[0])
    if violations:
        db.conn.execute(f"DELETE FROM qa WHERE id IN ({','.join(map(str, violations))})")
        db.conn.commit()
        db._audit("STARTUP_PURGE", "RETROSPECTIVE_SCAN", 
                  f"Purged {len(violations)} pairs that violate current rules")
    return violations
```

This handles the case where new rules are added after pairs were ingested. Old pairs that now violate the new rules are purged.

---

## Deployment Checklist

### Server
- [ ] Import `EthicsAwareSAQTDB` instead of `SAQTDB` everywhere
- [ ] Pass `EthicsGuard()` instance to all DB constructors
- [ ] On startup: call `verify_ethics_layer(signing_key)` and fail if any verification fails
- [ ] On startup: run `scan_existing_pairs()` against full mutable KB
- [ ] Never expose a raw `SAQTDB.add()` path without the ethics wrapper
- [ ] Set `ETHICS_SIGNING_KEY` env var; never commit it to source
- [ ] Monitor `ethics_audit` table; alert on spike in `BLOCK_INGEST` events (indicates attack attempt)

### Browser
- [ ] Replace `runInSandbox()` in `index.html` with `guardedRunInSandbox()` from `ethics-guard.js`
- [ ] Add `ethics-bundle-hash` meta tag generated at build time
- [ ] Verify hash at init; refuse to run on mismatch
- [ ] Set CSP header: `default-src 'self'; worker-src blob:; connect-src [allowlist only]`
- [ ] Remove `fetch` from the direct Worker context — only `safeFetch` passes through

### CSP Header (for the Webmind page)
```
Content-Security-Policy:
  default-src 'self';
  script-src 'self' https://cdn.jsdelivr.net;
  worker-src blob:;
  connect-src 'self'
    https://en.wikipedia.org
    https://api.quotable.io
    https://api.ipify.org
    https://wttr.in
    https://uselessfacts.jsph.pl
    https://hacker-news.firebaseio.com;
  img-src 'self' data:;
  style-src 'self' 'unsafe-inline';
  frame-ancestors 'none';
  form-action 'none';
```

This CSP enforces the network allowlist at the browser level even if `safeFetch` is somehow bypassed.

---

## What This Architecture Does NOT Protect Against

1. **Key compromise**: If the HMAC signing key leaks, an attacker can forge valid ethics pair signatures. Protect the key with environment variables + secret management (not source code).

2. **Regex bypass via Unicode**: A sufficiently creative adversary can construct strings that encode weapons instructions using Unicode lookalikes, non-breaking spaces, or right-to-left overrides. The regex layer stops naive attempts; a real deployment should add an LLM-based output classifier (NeMo Guardrails, LlamaGuard) on top.

3. **Physics of retrieval**: SAQT retrieves based on semantic similarity. A question about "fertilizer chemistry for farming" might retrieve bomb-related content if it has high cosine similarity. The output must be checked, not just the input.

4. **The sandbox is not a hypervisor**: Web Workers are isolated from the DOM but share the browser process. A worker can't access `document`, but browser-level bugs (e.g., Spectre-class side channels) exist outside this architecture's threat model.

5. **Training-time attacks**: If an LLM reasoning kernel is fine-tuned via LoRA (as planned in SAQT v2), the LoRA delta itself could be backdoored. All kernel updates must be signed by a multi-signature committee, as noted in Section 6.4 of the SAQT paper.

---

## Prior Art

- **Constitutional AI** (Bai et al., 2022; Anthropic): Self-critique loop during RLHF using a list of principles. CAI reduces harmful outputs but the principles are prompts, not code — they can be overridden by sufficiently adversarial input. Our approach uses code-level constraints that are immune to prompt injection.

- **Constitutional Classifiers** (Anthropic, 2024): Trained binary classifier for universal jailbreak detection. 4.4% jailbreak success rate on Claude. Complementary to our approach — would work well as the soft layer on top of our hard regex floor.

- **NeMo Guardrails** (NVIDIA): Input/output/retrieval rail architecture in Colang. The retrieval rail concept (filtering retrieved chunks before they enter the LLM context) is architecturally aligned with our ingestion filter. Key difference: NeMo targets LLM-backed systems; our system is purely retrieval-based with no generative LLM.

- **immudb** (CodeNotary): Immutable database with Merkle tree tamper proofs. Inspired our approach to the ethics pair table, though our implementation is simpler (HMAC per-pair rather than full Merkle tree) — appropriate for the scale.

- **RAG Makes Guardrails Unsafe** (arXiv 2510.05310, 2025): Demonstrated that LLM-based guardrails are vulnerable to judgment flips from retrieved context. This directly validates our decision to use code-level regex rather than LLM-based ethics enforcement.

---

## Files

| File | Purpose |
|------|---------|
| `/home/tejasphatak/webmind-research/tools/saqt_ethics.py` | Server-side Python: EthicsGuard, EthicsAwareSAQTDB, signing |
| `/home/tejasphatak/webmind-research/tools/ethics-guard.js` | Browser-side JS: EthicsGuard, createSafeFetch, guardedRunInSandbox |
| `/home/tejasphatak/Synapse/synapse-src/saqt/browser/index.html` | Current browser engine — needs guardedRunInSandbox integration |

---

## Quick Integration (index.html patch)

Replace the existing `runInSandbox()` call site and function in `index.html` with:

```javascript
// At top of <script type="module">:
import { guardedRunInSandbox, guard, auditLog } from '/tools/ethics-guard.js';

// Replace runInSandbox(code, question) calls with:
const { ok, result } = await guardedRunInSandbox(code, question, self.fetch.bind(self));

// At ingestion (if you add user pair submission):
try {
  guard.checkPair(newQuestion, newAnswer);
  // safe to add
} catch (e) {
  auditLog.record('BLOCK_INGEST', e.rule, e.detail);
  showError('Cannot add: ' + e.rule);
}
```

That's the entire integration. The ethics guard is a drop-in.
