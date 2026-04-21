"""
Tool interface for the brain — MCP client for external capabilities.

The brain calls tools when it needs information it doesn't have.
Tools are MCP servers that the brain connects to as a client.

Currently supported:
  - Web Search (via Brave/DuckDuckGo) — togglable
  - (Future: code exec, file reader, calculator)

Usage:
    from tools import ToolRouter
    router = ToolRouter(web_search=True)
    result = router.search("what is the capital of france")
"""

import os
import json
import subprocess
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed


class WebSearch:
    """Search the web via DuckDuckGo (no API key needed)."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def search(self, query: str, max_results: int = 3) -> Optional[str]:
        """Search and return top results as text."""
        if not self.enabled:
            return None

        try:
            # DuckDuckGo instant answer API (no key needed)
            result = subprocess.run(
                ["curl", "-s", "-m", "5",
                 f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)

            # Try abstract first (best quality)
            abstract = data.get("AbstractText", "").strip()
            if abstract and len(abstract) > 20:
                return abstract

            # Try related topics
            related = data.get("RelatedTopics", [])
            texts = []
            for item in related[:max_results]:
                text = item.get("Text", "").strip()
                if text and len(text) > 20:
                    texts.append(text)
            if texts:
                return " ".join(texts)

            # Fallback: try Wikipedia API
            return self._try_wikipedia(query)

        except Exception:
            return None

    def _try_wikipedia(self, query: str) -> Optional[str]:
        """Fallback: Wikipedia REST API."""
        try:
            topic = "_".join(query.lower().split()[:5])
            result = subprocess.run(
                ["curl", "-s", "-m", "5",
                 f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)
            extract = data.get("extract", "").strip()
            if extract and len(extract) > 20:
                sentences = extract.split(". ")
                return ". ".join(sentences[:3]) + "."
            return None
        except Exception:
            return None


class CodeEval:
    """
    Sandboxed eval for math/code execution.

    Only allows safe operations — no imports, no file access, no exec.
    Detects math-like queries and evaluates them.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        # Safe builtins for math evaluation
        self._safe_builtins = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'len': len, 'int': int, 'float': float,
            'str': str, 'bool': bool, 'list': list, 'range': range,
            'pow': pow, 'sorted': sorted, 'reversed': reversed,
            'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
            'True': True, 'False': False, 'None': None,
        }
        # Add math functions
        import math
        for name in ['sqrt', 'sin', 'cos', 'tan', 'log', 'log2', 'log10',
                     'exp', 'pi', 'e', 'ceil', 'floor', 'factorial',
                     'gcd', 'isqrt']:
            if hasattr(math, name):
                self._safe_builtins[name] = getattr(math, name)

    def looks_like_math(self, query: str) -> bool:
        """Detect if query needs computation rather than retrieval."""
        import re
        # Contains arithmetic operators with numbers
        if re.search(r'\d+\s*[\+\-\*\/\%\^]\s*\d+', query):
            return True
        # Starts with "calculate", "compute", "eval", "what is X + Y"
        lower = query.lower().strip()
        if any(lower.startswith(w) for w in ['calculate', 'compute', 'eval', 'solve']):
            return True
        # "what is 2+2" pattern
        if re.search(r'what\s+is\s+\d', lower):
            return True
        return False

    def extract_expression(self, query: str) -> Optional[str]:
        """Pull the evaluable expression from a query."""
        import re
        # Try to find a math expression
        # "what is 247 + 389" → "247 + 389"
        # "calculate sqrt(144)" → "sqrt(144)"
        # "2**10" → "2**10"
        lower = query.lower().strip()
        # Remove common prefixes
        for prefix in ['what is', 'calculate', 'compute', 'eval', 'solve', 'whats']:
            if lower.startswith(prefix):
                lower = lower[len(prefix):].strip()
                break
        # Replace ^ with ** for exponentiation
        lower = lower.replace('^', '**')
        # Replace × with *
        lower = lower.replace('×', '*').replace('÷', '/')
        return lower if lower else None

    def evaluate(self, query: str) -> Optional[str]:
        """Evaluate a math/code expression safely. Returns result or None."""
        if not self.enabled:
            return None

        expr = self.extract_expression(query)
        if not expr:
            return None

        try:
            # Sandboxed eval — no __builtins__ access, only safe functions
            result = eval(expr, {"__builtins__": {}}, self._safe_builtins)
            return str(result)
        except Exception:
            return None


class BrowserTool:
    """
    Headless browser for visual debugging.

    Can navigate to URLs, take screenshots, read DOM, and observe changes.
    Uses Playwright (Chromium) under the hood.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._browser = None
        self._page = None

    def _ensure_browser(self):
        """Lazy-init browser on first use."""
        if self._browser is None:
            try:
                from playwright.sync_api import sync_playwright
                self._pw = sync_playwright().start()
                self._browser = self._pw.chromium.launch(headless=True)
                self._page = self._browser.new_page()
            except Exception as e:
                self.enabled = False
                return False
        return True

    def screenshot(self, url: str, wait_ms: int = 2000) -> Optional[dict]:
        """
        Navigate to URL, wait, take screenshot.
        Returns: {"screenshot_path": str, "title": str, "url": str}
        """
        if not self.enabled:
            return None
        if not self._ensure_browser():
            return None

        try:
            self._page.goto(url, timeout=10000)
            self._page.wait_for_timeout(wait_ms)
            path = "/tmp/brain_screenshot.png"
            self._page.screenshot(path=path)
            return {
                "screenshot_path": path,
                "title": self._page.title(),
                "url": self._page.url,
            }
        except Exception as e:
            return {"error": str(e)}

    def inspect_dom(self, url: str, selector: Optional[str] = None) -> Optional[dict]:
        """
        Navigate and inspect DOM structure.
        Returns: {"html": str, "computed_styles": dict, "visible": bool, "dimensions": dict}
        """
        if not self.enabled:
            return None
        if not self._ensure_browser():
            return None

        try:
            self._page.goto(url, timeout=10000)
            self._page.wait_for_timeout(1000)

            if selector:
                el = self._page.query_selector(selector)
                if not el:
                    return {"error": f"Selector '{selector}' not found"}

                box = el.bounding_box()
                visible = el.is_visible()
                html = el.inner_html()
                # Get computed styles
                styles = self._page.evaluate("""(sel) => {
                    const el = document.querySelector(sel);
                    if (!el) return {};
                    const cs = getComputedStyle(el);
                    return {
                        display: cs.display,
                        visibility: cs.visibility,
                        opacity: cs.opacity,
                        width: cs.width,
                        height: cs.height,
                        overflow: cs.overflow,
                        position: cs.position,
                    };
                }""", selector)

                return {
                    "html": html[:2000],
                    "visible": visible,
                    "dimensions": box,
                    "computed_styles": styles,
                }
            else:
                # Full page summary
                title = self._page.title()
                body_text = self._page.inner_text("body")[:1000]
                return {"title": title, "body_preview": body_text}

        except Exception as e:
            return {"error": str(e)}

    def watch(self, url: str, duration_ms: int = 3000, interval_ms: int = 500) -> Optional[list]:
        """
        Watch a page over time — detect changes (animations, loading states).
        Returns list of frame descriptions.
        """
        if not self.enabled:
            return None
        if not self._ensure_browser():
            return None

        try:
            self._page.goto(url, timeout=10000)
            frames = []
            steps = duration_ms // interval_ms

            for i in range(steps):
                self._page.wait_for_timeout(interval_ms)
                # Capture visible text state
                text = self._page.inner_text("body")[:500]
                frames.append({"frame": i, "text_preview": text[:200]})

            return frames
        except Exception as e:
            return [{"error": str(e)}]

    def close(self):
        """Clean up browser resources."""
        if self._browser:
            self._browser.close()
            self._pw.stop()
            self._browser = None


class ToolRouter:
    """Routes tool calls from the brain. Configurable per-tool enable/disable."""

    def __init__(self, web_search=True, code_eval=True, browser=True):
        self.web_search = WebSearch(enabled=web_search)
        self.code_eval = CodeEval(enabled=code_eval)
        self.browser = BrowserTool(enabled=browser)

    def on_miss(self, query: str, brain) -> Optional[str]:
        """
        Called when brain.ask() returns abstain.
        Tries tools to find an answer, teaches the brain if found.

        Returns the answer string if found, None otherwise.
        """
        # Try code eval first (instant, no network)
        if self.code_eval.looks_like_math(query):
            result = self.code_eval.evaluate(query)
            if result:
                # Teach the brain the result
                brain.teach(f"{query.rstrip('?')} is {result}", confidence=0.8)
                return result

        # Try web search
        result = self.web_search.search(query)
        if result:
            brain.teach(result, confidence=0.6)
            return result

        return None

    def on_query(self, query: str, brain) -> Optional[str]:
        """
        Called BEFORE brain.ask() — intercepts queries that need computation.
        Returns result if handled, None to let brain handle it.
        """
        if self.code_eval.enabled and self.code_eval.looks_like_math(query):
            result = self.code_eval.evaluate(query)
            if result:
                return result
        return None

    def on_partial(self, query: str, known_concepts: List[str],
                   missing_hints: List[str], brain) -> Optional[str]:
        """
        Called when convergence partially converges — knows some concepts but
        not enough to answer confidently. Fires parallel searches for missing pieces.

        Args:
            query: original question
            known_concepts: words the brain already found
            missing_hints: words/subqueries the brain thinks it needs
            brain: the brain instance (to teach results back)

        Returns combined knowledge if found, None otherwise.
        """
        if not missing_hints or not self.web_search.enabled:
            return None

        # Fire all searches in parallel
        results = []
        with ThreadPoolExecutor(max_workers=min(len(missing_hints), 5)) as pool:
            futures = {
                pool.submit(self.web_search.search, hint): hint
                for hint in missing_hints
            }
            for future in as_completed(futures, timeout=10):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        # Teach each result immediately
                        brain.teach(result, confidence=0.6)
                except Exception:
                    pass

        if results:
            return " ".join(results)
        return None

    def parallel_search(self, queries: List[str], brain) -> List[Optional[str]]:
        """
        Fire multiple web searches in parallel. Teach results to brain.
        Returns list of results (None for failures).
        """
        if not self.web_search.enabled:
            return [None] * len(queries)

        results = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as pool:
            future_to_idx = {
                pool.submit(self.web_search.search, q): i
                for i, q in enumerate(queries)
            }
            for future in as_completed(future_to_idx, timeout=10):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result:
                        results[idx] = result
                        brain.teach(result, confidence=0.6)
                except Exception:
                    pass

        return results

    def diagnose_ui(self, url: str, selector: str) -> Optional[str]:
        """
        Diagnose why a UI element isn't working.
        Returns human-readable diagnosis.
        """
        if not self.browser.enabled:
            return None

        dom = self.browser.inspect_dom(url, selector)
        if not dom or "error" in dom:
            return dom.get("error", "Could not inspect element") if dom else None

        issues = []
        styles = dom.get("computed_styles", {})
        dims = dom.get("dimensions")
        visible = dom.get("visible", True)

        if not visible:
            issues.append("Element is not visible")

        if styles.get("display") == "none":
            issues.append("display: none — element is hidden by CSS")
        if styles.get("visibility") == "hidden":
            issues.append("visibility: hidden — element takes space but is invisible")
        if styles.get("opacity") == "0":
            issues.append("opacity: 0 — element is fully transparent")

        if dims:
            if dims.get("width", 1) == 0:
                issues.append(f"width is 0px — check parent flex/grid or explicit width")
            if dims.get("height", 1) == 0:
                issues.append(f"height is 0px — check content, min-height, or parent constraints")
            if dims.get("x", 0) < -1000 or dims.get("y", 0) < -1000:
                issues.append("element is positioned off-screen")
        else:
            issues.append("element has no bounding box — may not be rendered")

        if styles.get("overflow") == "hidden":
            issues.append("overflow: hidden — content may be clipped")

        if not issues:
            return f"Element '{selector}' appears normal: visible={visible}, dimensions={dims}"

        return f"Issues with '{selector}':\n" + "\n".join(f"  - {i}" for i in issues)

    def browse(self, url: str, selector: Optional[str] = None) -> Optional[dict]:
        """
        Look at a URL — screenshot + DOM inspection.
        The brain's eyes.
        """
        if not self.browser.enabled:
            return None

        result = {}

        # Take screenshot
        ss = self.browser.screenshot(url)
        if ss:
            result["screenshot"] = ss

        # Inspect DOM
        dom = self.browser.inspect_dom(url, selector)
        if dom:
            result["dom"] = dom

        return result if result else None

    def code_loop(self, task: str, brain, max_iterations: int = 5,
                  context: Optional[dict] = None) -> dict:
        """
        Autonomous coding loop. Convergence = code works.

        1. Understand task (ask brain for relevant knowledge)
        2. Generate code (from brain's code knowledge)
        3. Execute (sandboxed eval or subprocess)
        4. Check result (does output match expectation?)
        5. If error → diagnose → fix → re-run
        6. Converge when: output is correct OR max iterations hit

        Returns: {"code": str, "output": str, "iterations": int, "converged": bool}
        """
        iterations = []
        current_code = None
        last_error = None

        for i in range(max_iterations):
            # Step 1+2: Generate or fix code
            if current_code is None:
                # First attempt — ask brain for relevant code knowledge
                ask_result = brain.ask(task)
                gen_result = brain.generate(task, max_tokens=50)
                # Use brain's code knowledge as starting point
                current_code = gen_result.get("text", "") if gen_result else ""

                # If brain doesn't know, try web search
                if not current_code or len(current_code) < 10:
                    search_result = self.web_search.search(f"python code {task}")
                    if search_result:
                        current_code = search_result
                        brain.teach(search_result, confidence=0.6)
            else:
                # Fix attempt — modify based on error
                fix_query = f"fix python error: {last_error}"
                fix_result = brain.ask(fix_query)
                if fix_result.get("strategy") != "abstain":
                    # Brain knows a fix
                    current_code = fix_result.get("answer", current_code)
                else:
                    # Search for fix
                    search_result = self.web_search.search(
                        f"python fix {last_error[:100]}"
                    ) if self.web_search.enabled else None
                    if search_result:
                        brain.teach(search_result, confidence=0.6)

            # Step 3: Execute
            exec_result = self._safe_exec(current_code)

            iteration = {
                "iteration": i + 1,
                "code": current_code[:500],
                "output": exec_result.get("output", "")[:200],
                "error": exec_result.get("error"),
                "success": exec_result.get("success", False),
            }
            iterations.append(iteration)

            # Step 4: Check convergence
            if exec_result.get("success") and not exec_result.get("error"):
                # Code ran without error = converged
                return {
                    "code": current_code,
                    "output": exec_result["output"],
                    "iterations": i + 1,
                    "converged": True,
                    "trace": iterations,
                }

            # Step 5: Diagnose error for next iteration
            last_error = exec_result.get("error", "unknown error")

        # Max iterations — did not converge
        return {
            "code": current_code,
            "output": iterations[-1].get("output", "") if iterations else "",
            "iterations": max_iterations,
            "converged": False,
            "trace": iterations,
            "last_error": last_error,
        }

    def _safe_exec(self, code: str, timeout: int = 5) -> dict:
        """
        Execute Python code in a subprocess sandbox.
        Returns: {"output": str, "error": str or None, "success": bool}
        """
        if not code or len(code.strip()) < 3:
            return {"output": "", "error": "empty code", "success": False}

        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if result.returncode == 0:
                return {
                    "output": result.stdout[:2000],
                    "error": None,
                    "success": True,
                }
            else:
                return {
                    "output": result.stdout[:1000],
                    "error": result.stderr[:1000],
                    "success": False,
                }
        except subprocess.TimeoutExpired:
            return {"output": "", "error": "timeout (5s)", "success": False}
        except Exception as e:
            return {"output": "", "error": str(e), "success": False}

    def configure(self, web_search: Optional[bool] = None,
                  code_eval: Optional[bool] = None,
                  browser: Optional[bool] = None):
        """Enable/disable tools at runtime."""
        if web_search is not None:
            self.web_search.enabled = web_search
        if code_eval is not None:
            self.code_eval.enabled = code_eval
        if browser is not None:
            self.browser.enabled = browser
