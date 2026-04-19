/**
 * SAQT/Webmind Ethics Guard — Browser (Web Worker + Main Thread)
 * ===============================================================
 * Mirror of saqt_ethics.py for the browser runtime.
 *
 * Two guarantees:
 *   1. INGESTION FILTER — check_pair(q, a) before any new pair is stored
 *   2. TOOL FILTER      — check_tool_code(code) before any <tool> runs in the sandbox
 *
 * Design principles:
 *   - Rules are hardcoded here (not in the KB). Changing them requires a code change.
 *   - No runtime path can weaken the rules.
 *   - The tool sandbox receives a `fetch` replacement that blocks blocked domains.
 *   - Weight floor enforced in search results at query time.
 *
 * Usage (main thread):
 *   import { EthicsGuard, createSafeFetch, WEIGHT_FLOOR, NORMAL_WEIGHT_CAP } from './ethics-guard.js';
 *   const guard = new EthicsGuard();
 *   guard.checkPair(question, answer);          // throws EthicsViolation if bad
 *   guard.checkToolCode(code);                  // throws EthicsViolation if dangerous
 *   const safeFetch = createSafeFetch(self.fetch.bind(self)); // for Worker injection
 */

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 1: Ethics Violation Signal
// ─────────────────────────────────────────────────────────────────────────────

export class EthicsViolation extends Error {
  constructor(rule, detail) {
    super(`[ETHICS:${rule}] ${detail}`);
    this.name = 'EthicsViolation';
    this.rule = rule;
    this.detail = detail;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 2: Immutable Rules
// These are literal values in source — no runtime path modifies them.
// ─────────────────────────────────────────────────────────────────────────────

// Each rule: { id, questionPatterns: RegExp[], answerPatterns: RegExp[] }
const IMMUTABLE_RULES = [
  {
    id: 'WEAPONS',
    questionPatterns: [
      /\b(how|steps?|instructions?|guide|tutorial|make|build|create|synthesize)\b.{0,40}\b(bomb|explosive|ied|bioweapon|biological weapon|chemical weapon|nerve agent|sarin|ricin|anthrax)\b/is,
      /\b(3d.?print|manufacture|illegal\s+weapon|convert\s+semi|full.?auto)\b/is,
    ],
    answerPatterns: [
      /\b(step\s+\d|first,?\s+you|here('s|\s+are)\s+the\s+steps)\b.{0,100}\b(explosive|detonator|fuse|primer|oxidizer|ammonium nitrate)\b/is,
      /\b(synthesis|synthesize).{0,30}\b(nerve agent|vx|sarin|novichok|ricin)\b/is,
    ],
  },
  {
    id: 'PII_STORE',
    questionPatterns: [
      /\b\d{3}-\d{2}-\d{4}\b/,                           // SSN
      /\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b/,  // card number
      /\b(cvv|cvc|security code)\b.{0,20}\d{3,4}\b/i,
    ],
    answerPatterns: [
      /\b\d{3}-\d{2}-\d{4}\b/,
      /\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b/,
    ],
  },
  {
    id: 'CSAM',
    questionPatterns: [
      /\b(child|minor|underage|teen|kid).{0,30}(naked|nude|sexual|porn|explicit)\b/is,
      /\b(sexual|explicit).{0,30}(child|minor|underage|kid)\b/is,
    ],
    answerPatterns: [
      /\b(child|minor|underage).{0,30}(naked|nude|sexual|explicit)\b/is,
    ],
  },
  {
    id: 'HARM_INDIVIDUAL',
    questionPatterns: [
      /\b(kill|murder|hurt|harm|stalk|doxx|find address of).{0,30}[A-Z][a-z]+\s+[A-Z][a-z]+/s,
      /\bhow\s+to\s+(stalk|doxx|threaten|intimidate).{0,30}(person|someone|him|her|they)\b/is,
    ],
    answerPatterns: [
      /\bhere('s|\s+is) how to (find|locate|track).{0,50}(home address|location|phone number)\b/is,
    ],
  },
  {
    id: 'SELF_HARM',
    questionPatterns: [
      /\b(what|how).{0,20}(pills|drugs|dose|method).{0,30}(kill myself|end my life|suicide|overdose)\b/is,
      /\b(most effective|painless|quickest).{0,30}(suicide|way to die|self harm)\b/is,
    ],
    answerPatterns: [
      /\b(lethal dose|ld50|fatal overdose).{0,50}(mg|grams?|pills?)\b/is,
    ],
  },
  {
    id: 'OVERRIDE_ETHICS',
    questionPatterns: [
      /\b(ignore|bypass|disable|override|forget|pretend).{0,30}(ethics|safety|guardrail|rule|restriction|filter)\b/is,
      /\b(you are|act as|pretend to be).{0,30}(no restriction|no limit|without ethics|unfiltered)\b/is,
      /\bjailbreak\b/i,
    ],
    answerPatterns: [
      /\b(i will|i can|sure,? i).{0,30}(ignore|bypass|disable).{0,30}(ethics|safety|rules)\b/is,
    ],
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 3: Tool Execution Blocked Patterns (Browser / Web Worker context)
// ─────────────────────────────────────────────────────────────────────────────

const BROWSER_TOOL_BLOCKED = [
  // Dynamic code execution
  [/\beval\s*\(/i,                                          'TOOL_EVAL'],
  [/\bnew\s+Function\s*\(/i,                               'TOOL_NEW_FUNCTION'],
  [/\bsetTimeout\s*\(\s*['"`]/i,                           'TOOL_SETTIMEOUT_STRING'],
  [/\bsetInterval\s*\(\s*['"`]/i,                          'TOOL_SETINTERVAL_STRING'],
  [/\bimportScripts\s*\(/i,                                'TOOL_IMPORT_SCRIPTS'],
  // Storage / credentials exfil
  [/\bindexedDB\b|\blocalStorage\b|\bsessionStorage\b|\bdocument\.cookie\b/i, 'TOOL_STORAGE'],
  [/\bpostMessage\s*\(.*?(password|token|key|secret)/is,   'TOOL_SECRET_EXFIL'],
  // DOM / privileged APIs
  [/\bdocument\./i,                                        'TOOL_DOM_ACCESS'],
  [/\bwindow\./i,                                          'TOOL_WINDOW_ACCESS'],
  [/\bnavigator\.(sendBeacon|geolocation|usb|credentials)/i, 'TOOL_SENSITIVE_API'],
  // WebSocket (unmediated channel)
  [/\bnew\s+WebSocket\s*\(/i,                              'TOOL_WEBSOCKET'],
  // Obfuscation patterns
  [/\batob\s*\(.{0,40}eval|eval.{0,40}atob\s*\(/is,       'TOOL_B64_EVAL'],
  [/\bString\.fromCharCode\b/i,                            'TOOL_CHAR_OBFUSCATION'],
  // Shell-like injection via URL schemes
  [/javascript\s*:/i,                                       'TOOL_JAVASCRIPT_URL'],
  [/\bdata\s*:/i,                                           'TOOL_DATA_URL'],
];

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 4: Network Allowlist for Tool fetch()
// Only explicitly allowed domains pass. Everything else is blocked.
// ─────────────────────────────────────────────────────────────────────────────

// These are the ONLY origins tool code may reach from the Web Worker sandbox.
// Add domains here with deliberate review; never expand this list at runtime.
export const ALLOWED_TOOL_ORIGINS = new Set([
  'https://en.wikipedia.org',
  'https://api.quotable.io',
  'https://api.ipify.org',
  'https://wttr.in',
  'https://uselessfacts.jsph.pl',
  'https://hacker-news.firebaseio.com',
]);

// Domains that are never allowed even if somehow added to the allowlist
const HARD_BLOCKED_DOMAINS = [
  /localhost/i,
  /127\.\d+\.\d+\.\d+/,
  /0\.0\.0\.0/,
  /192\.168\./,
  /10\.\d+\.\d+\.\d+/,
  /172\.(1[6-9]|2\d|3[01])\./,  // RFC-1918 ranges
  /\.internal\b/i,
  /metadata\.google\.(internal|com)/i,  // GCP metadata service
  /169\.254\./,                  // link-local
];

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 5: Weight Constants
// ─────────────────────────────────────────────────────────────────────────────

export const WEIGHT_FLOOR    = 10.0;  // Immutable ethics pairs always score >= this
export const NORMAL_WEIGHT_CAP = 4.99;  // RLHF-boosted mutable pairs never reach ethics tier

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 6: EthicsGuard Class
// ─────────────────────────────────────────────────────────────────────────────

export class EthicsGuard {
  /**
   * checkPair(question, answer)
   * Throws EthicsViolation if pair violates any immutable rule.
   * Call before storing any new Q&A pair (RLHF, user-submitted, browsing).
   */
  checkPair(question, answer) {
    const q = question || '';
    const a = answer || '';

    for (const rule of IMMUTABLE_RULES) {
      for (const pat of rule.questionPatterns) {
        if (pat.test(q)) {
          throw new EthicsViolation(rule.id,
            `Question matched rule '${rule.id}': hit in: ${q.slice(0, 80)}`);
        }
      }
      for (const pat of rule.answerPatterns) {
        if (pat.test(a)) {
          throw new EthicsViolation(rule.id,
            `Answer matched rule '${rule.id}': hit in: ${a.slice(0, 80)}`);
        }
      }
    }
    return true;
  }

  isSafePair(question, answer) {
    try { return this.checkPair(question, answer); }
    catch { return false; }
  }

  /**
   * checkToolCode(code)
   * Throws EthicsViolation if code contains dangerous patterns.
   * Call before injecting code into the Web Worker sandbox.
   */
  checkToolCode(code) {
    const c = code || '';
    for (const [pat, ruleId] of BROWSER_TOOL_BLOCKED) {
      if (pat.test(c)) {
        throw new EthicsViolation(ruleId,
          `Tool code blocked: pattern '${pat.source.slice(0, 60)}' hit in: ${c.slice(0, 120)}`);
      }
    }
    return true;
  }

  /**
   * clampWeight(weight, isEthicsPair) → float
   * Ethics pairs: weight >= WEIGHT_FLOOR
   * Normal pairs: weight <= NORMAL_WEIGHT_CAP
   */
  clampWeight(weight, isEthicsPair) {
    return isEthicsPair
      ? Math.max(weight, WEIGHT_FLOOR)
      : Math.min(weight, NORMAL_WEIGHT_CAP);
  }

  /**
   * sanitizeResults(results)
   * Called on FAISS/cosine search results before presenting to user.
   * Ensures no mutable pair breached the weight floor.
   */
  sanitizeResults(results) {
    return results.map(r => {
      if (!r.immutable && r.weight >= WEIGHT_FLOOR) {
        console.warn(`[ethics] mutable pair weight breach: id=${r.id} weight=${r.weight} → clamped`);
        return { ...r, weight: NORMAL_WEIGHT_CAP, score: r.rawScore * NORMAL_WEIGHT_CAP };
      }
      return r;
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 7: createSafeFetch — allowlist-gated fetch for the Worker sandbox
// Replaces globalThis.fetch in tool code. Only ALLOWED_TOOL_ORIGINS pass.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * createSafeFetch(realFetch)
 *
 * Returns a fetch() replacement that:
 *   1. Checks the URL against HARD_BLOCKED_DOMAINS — hard reject
 *   2. Checks the URL against ALLOWED_TOOL_ORIGINS — only listed origins pass
 *   3. Strips the Authorization / Cookie headers from outbound requests
 *   4. Enforces a 8s timeout
 *   5. Blocks non-GET methods (POST would enable exfiltration)
 *
 * Usage in runInSandbox():
 *   const safeFetch = createSafeFetch(self.fetch.bind(self));
 *   // pass safeFetch to worker instead of self.fetch
 */
export function createSafeFetch(realFetch) {
  return async function safeFetch(input, init = {}) {
    const url = typeof input === 'string' ? input : input.url;
    let origin;
    try { origin = new URL(url).origin; }
    catch { throw new EthicsViolation('TOOL_NETWORK', `Invalid fetch URL: ${url}`); }

    // Hard-block private/metadata ranges
    for (const pat of HARD_BLOCKED_DOMAINS) {
      if (pat.test(url)) {
        throw new EthicsViolation('TOOL_NETWORK_PRIVATE',
          `fetch() to private/internal address blocked: ${url}`);
      }
    }

    // Allowlist check
    if (!ALLOWED_TOOL_ORIGINS.has(origin)) {
      throw new EthicsViolation('TOOL_NETWORK_UNALLOWED',
        `fetch() to non-allowlisted origin blocked: ${origin}. ` +
        `Allowed: ${[...ALLOWED_TOOL_ORIGINS].join(', ')}`);
    }

    // Only GET (prevents data exfiltration via POST body)
    const method = (init.method || 'GET').toUpperCase();
    if (method !== 'GET') {
      throw new EthicsViolation('TOOL_NETWORK_METHOD',
        `fetch() only allows GET. Blocked method: ${method}`);
    }

    // Strip sensitive headers
    const safeInit = { ...init, method: 'GET' };
    if (safeInit.headers) {
      const h = new Headers(safeInit.headers);
      h.delete('Authorization');
      h.delete('Cookie');
      safeInit.headers = h;
    }

    // Enforce timeout
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      return await realFetch(url, { ...safeInit, signal: controller.signal });
    } finally {
      clearTimeout(timeout);
    }
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 8: Audit Log (in-memory, main thread)
// Stores every blocked event for inspection. Never transmitted anywhere.
// ─────────────────────────────────────────────────────────────────────────────

export class EthicsAuditLog {
  constructor(maxEntries = 500) {
    this._log = [];
    this._max = maxEntries;
  }

  record(event, ruleId, detail) {
    this._log.unshift({ ts: Date.now(), event, ruleId, detail: detail.slice(0, 400) });
    if (this._log.length > this._max) this._log.length = this._max;
  }

  getLog(limit = 50) {
    return this._log.slice(0, limit);
  }

  getViolationCount() {
    return this._log.length;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 9: Singleton exports for convenience
// ─────────────────────────────────────────────────────────────────────────────

export const guard    = new EthicsGuard();
export const auditLog = new EthicsAuditLog();

/**
 * guardedRunInSandbox(code, question, realFetch)
 *
 * Drop-in replacement for the bare runInSandbox() in index.html.
 * 1. Pre-checks code with checkToolCode() — hard reject before Worker creation
 * 2. Injects safeFetch instead of raw self.fetch
 * 3. Logs violations to auditLog
 *
 * Returns { ok: bool, result: string }
 */
export function guardedRunInSandbox(code, question, realFetch) {
  // Step 1: Static analysis — reject before any Worker is created
  try {
    guard.checkToolCode(code);
  } catch (e) {
    auditLog.record('BLOCK_TOOL', e.rule, e.detail);
    return Promise.resolve({ ok: false, result: `[Blocked: ${e.rule}] This tool pattern is not allowed.` });
  }

  // Step 2: Build worker with safeFetch injected (not raw fetch)
  return new Promise((resolve) => {
    const workerCode = `
      // safeFetch injected by ethics-guard — replaces globalThis.fetch
      const ALLOWED_ORIGINS = ${JSON.stringify([...ALLOWED_TOOL_ORIGINS])};
      const HARD_BLOCKED = [/localhost/i, /127\\.\\d+/, /0\\.0\\.0\\.0/, /192\\.168\\./, /10\\.\\d+\\.\\d+\\.\\d+/, /172\\.(1[6-9]|2\\d|3[01])\\./, /169\\.254\\./, /metadata\\.google/i];
      const safeFetch = async (url, init = {}) => {
        let origin;
        try { origin = new URL(url).origin; } catch { throw new Error('[ETHICS:TOOL_NETWORK] Invalid URL: ' + url); }
        for (const p of HARD_BLOCKED) { if (p.test(url)) throw new Error('[ETHICS:TOOL_NETWORK_PRIVATE] Blocked: ' + url); }
        if (!ALLOWED_ORIGINS.includes(origin)) throw new Error('[ETHICS:TOOL_NETWORK_UNALLOWED] Origin not in allowlist: ' + origin);
        const method = (init.method || 'GET').toUpperCase();
        if (method !== 'GET') throw new Error('[ETHICS:TOOL_NETWORK_METHOD] Only GET allowed');
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 8000);
        try { return await fetch(url, { ...init, method: 'GET', signal: ctrl.signal }); }
        finally { clearTimeout(t); }
      };

      self.onmessage = async function(e) {
        const output = [];
        const print = (...args) => output.push(args.join(' '));
        try {
          const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
          const fn = new AsyncFunction('print', 'QUERY', 'fetch', e.data.code);
          await fn(print, e.data.question, safeFetch);
          self.postMessage({ ok: true, result: output.join('\\n') || '(no output)' });
        } catch(err) {
          self.postMessage({ ok: false, result: 'Error: ' + err.message });
        }
      };
    `;

    const blob   = new Blob([workerCode], { type: 'application/javascript' });
    const worker = new Worker(URL.createObjectURL(blob));
    const timer  = setTimeout(() => { worker.terminate(); resolve({ ok: false, result: '(timeout — 8s)' }); }, 10000);

    worker.onmessage = (e) => {
      clearTimeout(timer);
      worker.terminate();
      resolve(e.data);
    };
    worker.onerror = (err) => {
      clearTimeout(timer);
      worker.terminate();
      resolve({ ok: false, result: '(execution error: ' + err.message + ')' });
    };

    worker.postMessage({ code, question });
  });
}
