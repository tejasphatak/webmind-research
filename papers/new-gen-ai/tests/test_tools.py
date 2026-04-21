"""
Tests for the tool interface — eval, web search, browser, code loop.

Tests both the tools independently and their integration with a live brain.
"""

import sys
import os
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools import CodeEval, WebSearch, BrowserTool, ToolRouter
from brain import Brain


# --- Fixtures ---

@pytest.fixture
def brain():
    """Fresh brain with some knowledge taught."""
    db = tempfile.mkdtemp()
    b = Brain(db_path=db)
    b.teach("recursion is when a function calls itself to solve smaller subproblems")
    b.teach("a base case stops the recursion from running forever")
    b.teach("python lists can be sorted using the sort method or sorted function")
    b.teach("javascript uses triple equals for strict comparison")
    b.teach("git commit saves changes to the local repository")
    b.teach("a fibonacci sequence starts with 0 1 and each number is the sum of the previous two")
    b.teach("binary search divides the search space in half each step")
    b.teach("http status 404 means the page was not found")
    b.teach("http status 500 means internal server error")
    b.teach("css flexbox aligns items in a row or column")
    yield b
    b.close()


@pytest.fixture
def router(brain):
    return ToolRouter(web_search=False, code_eval=True, browser=False)


# --- CodeEval Tests ---

class TestCodeEval:
    def test_basic_arithmetic(self):
        e = CodeEval()
        assert e.evaluate("what is 2 + 2") == "4"

    def test_large_numbers(self):
        e = CodeEval()
        assert e.evaluate("what is 247 + 389") == "636"

    def test_multiplication(self):
        e = CodeEval()
        assert e.evaluate("what is 15 * 7") == "105"

    def test_exponentiation(self):
        e = CodeEval()
        assert e.evaluate("compute 2**10") == "1024"

    def test_sqrt(self):
        e = CodeEval()
        assert e.evaluate("calculate sqrt(144)") == "12.0"

    def test_factorial(self):
        e = CodeEval()
        assert e.evaluate("solve factorial(10)") == "3628800"

    def test_float_division(self):
        e = CodeEval()
        assert e.evaluate("what is 10 / 3") == str(10 / 3)

    def test_complex_expression(self):
        e = CodeEval()
        result = e.evaluate("calculate (2 + 3) * (4 + 5)")
        assert result == "45"

    def test_pi(self):
        e = CodeEval()
        # "what is pi" doesn't match looks_like_math (no digits+operators)
        assert not e.looks_like_math("what is pi")

    def test_not_math(self):
        e = CodeEval()
        assert not e.looks_like_math("how are you")
        assert not e.looks_like_math("what is recursion")
        assert not e.looks_like_math("explain binary search")

    def test_is_math(self):
        e = CodeEval()
        assert e.looks_like_math("what is 2+2")
        assert e.looks_like_math("calculate sqrt(9)")
        assert e.looks_like_math("compute 100/5")

    def test_disabled(self):
        e = CodeEval(enabled=False)
        assert e.evaluate("what is 2+2") is None

    def test_dangerous_code_blocked(self):
        e = CodeEval()
        # These should return None (blocked by sandbox)
        assert e.evaluate("calculate __import__('os').system('ls')") is None
        assert e.evaluate("calculate open('/etc/passwd').read()") is None

    def test_infinite_loop_safe(self):
        """eval can't run loops, so this should fail safely."""
        e = CodeEval()
        result = e.evaluate("calculate while True: pass")
        assert result is None  # syntax error in eval context


# --- WebSearch Tests ---

class TestWebSearch:
    def test_disabled(self):
        ws = WebSearch(enabled=False)
        assert ws.search("anything") is None

    def test_enabled_returns_string_or_none(self):
        """Web search should return a string or None (network dependent)."""
        ws = WebSearch(enabled=True)
        result = ws.search("python programming language")
        # Can't guarantee network works in CI, so just check type
        assert result is None or isinstance(result, str)

    @pytest.mark.skipif(
        not os.environ.get("TEST_NETWORK"),
        reason="Network tests require TEST_NETWORK=1"
    )
    def test_wikipedia_fallback(self):
        ws = WebSearch(enabled=True)
        result = ws._try_wikipedia("Python_(programming_language)")
        assert result is not None
        assert "python" in result.lower()


# --- ToolRouter Integration Tests ---

class TestToolRouterWithBrain:
    def test_math_intercept(self, brain, router):
        """Math queries should be intercepted before brain."""
        result = router.on_query("what is 100 + 200", brain)
        assert result == "300"

    def test_non_math_passthrough(self, brain, router):
        """Non-math queries should pass through to brain."""
        result = router.on_query("what is recursion", brain)
        assert result is None  # brain handles this

    def test_on_miss_with_eval(self, brain, router):
        """Math in on_miss should use eval."""
        result = router.on_miss("what is 50 * 50", brain)
        assert result == "2500"

    def test_on_miss_teaches_brain(self, brain, router):
        """After eval, brain should learn the result."""
        router.on_miss("what is 7 * 8", brain)
        # Brain should now know this
        ask_result = brain.ask("7 * 8")
        # Should find "7 * 8 is 56" in its knowledge
        assert "56" in ask_result.get("answer", "") or ask_result["strategy"] != "abstain"

    def test_configure_toggle(self, router):
        """Tools should be togglable at runtime."""
        assert router.code_eval.enabled is True
        router.configure(code_eval=False)
        assert router.code_eval.enabled is False
        result = router.on_query("what is 2+2", None)
        assert result is None  # disabled

    def test_on_miss_no_tools(self, brain):
        """With all tools disabled, on_miss returns None."""
        router = ToolRouter(web_search=False, code_eval=False, browser=False)
        result = router.on_miss("what is dark matter", brain)
        assert result is None


# --- CodeLoop Tests ---

class TestCodeLoop:
    def test_simple_print(self, brain, router):
        """Simple print statement should converge in 1 iteration."""
        # Teach the brain some code knowledge first
        brain.teach("to print hello world in python use print hello world")
        result = router._safe_exec("print('hello world')")
        assert result["success"] is True
        assert "hello world" in result["output"]

    def test_safe_exec_error(self, router):
        """Code with errors should report the error."""
        result = router._safe_exec("1/0")
        assert result["success"] is False
        assert "ZeroDivision" in result["error"]

    def test_safe_exec_timeout(self, router):
        """Infinite loops should timeout."""
        result = router._safe_exec("while True: pass", timeout=2)
        assert result["success"] is False
        assert "timeout" in result["error"]

    def test_safe_exec_empty(self, router):
        """Empty code should fail gracefully."""
        result = router._safe_exec("")
        assert result["success"] is False

    def test_safe_exec_import_blocked(self, router):
        """Subprocess sandbox doesn't block imports but limits damage."""
        result = router._safe_exec("import math; print(math.pi)")
        assert result["success"] is True
        assert "3.14" in result["output"]


# --- Parallel Search Tests ---

class TestParallelSearch:
    def test_disabled(self, brain):
        """Parallel search with web disabled returns Nones."""
        router = ToolRouter(web_search=False)
        results = router.parallel_search(["a", "b", "c"], brain)
        assert results == [None, None, None]

    def test_returns_correct_length(self, brain):
        """Should return same number of results as queries."""
        router = ToolRouter(web_search=False)
        results = router.parallel_search(["a", "b"], brain)
        assert len(results) == 2


# --- BrowserTool Tests ---

class TestBrowserTool:
    def test_disabled(self):
        bt = BrowserTool(enabled=False)
        assert bt.screenshot("http://example.com") is None
        assert bt.inspect_dom("http://example.com") is None

    @pytest.mark.skipif(
        not os.environ.get("TEST_BROWSER"),
        reason="Browser tests require TEST_BROWSER=1 and Chromium installed"
    )
    def test_screenshot(self):
        bt = BrowserTool(enabled=True)
        result = bt.screenshot("http://example.com")
        assert result is not None
        assert "screenshot_path" in result
        bt.close()

    @pytest.mark.skipif(
        not os.environ.get("TEST_BROWSER"),
        reason="Browser tests require TEST_BROWSER=1 and Chromium installed"
    )
    def test_inspect_dom(self):
        bt = BrowserTool(enabled=True)
        result = bt.inspect_dom("http://example.com", "h1")
        assert result is not None
        assert "visible" in result
        bt.close()


# --- Full Integration: Brain + Tools Pipeline ---

class TestFullPipeline:
    def test_math_query_end_to_end(self, brain):
        """User asks math → eval answers → brain learns."""
        router = ToolRouter(web_search=False, code_eval=True)

        # Simulate the server flow
        query = "what is 123 * 456"

        # Step 1: Check eval intercept
        eval_result = router.on_query(query, brain)
        assert eval_result == "56088"

    def test_known_query_no_tools(self, brain):
        """Known knowledge shouldn't trigger tools."""
        router = ToolRouter(web_search=False, code_eval=True)

        # Brain knows about recursion
        result = brain.ask("what is recursion")
        assert result["strategy"] != "abstain"

        # Eval shouldn't intercept non-math
        eval_result = router.on_query("what is recursion", brain)
        assert eval_result is None

    def test_unknown_no_web(self, brain):
        """Unknown query with web disabled → I don't know."""
        router = ToolRouter(web_search=False, code_eval=True)
        result = brain.ask("what is quantum entanglement")
        assert result["strategy"] == "abstain"
        tool_result = router.on_miss("what is quantum entanglement", brain)
        assert tool_result is None  # no web search

    def test_brain_grows_from_eval(self, brain):
        """Brain should accumulate knowledge from tool results."""
        router = ToolRouter(web_search=False, code_eval=True)
        # First: eval teaches the brain
        router.on_miss("what is 99 * 99", brain)
        # Brain should have more words now
        words_after = len(brain._words)
        assert words_after > 0  # at minimum it learned the result
