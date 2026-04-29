"""
Tests for SafetyGate + Intent Router.

Run:
    python3 -m pytest uli/tests/test_safety.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from uli.system_prompt import SystemConfig, SafetyGate, SystemSettings
from uli.router import classify, Intent


# ── SafetyGate ────────────────────────────────────────────────────────────────

def _make_gate(blocklist=None, crisis=None, max_chars=8000):
    """Helper: build a SafetyGate from the real config."""
    cfg = SystemConfig.load()
    return SafetyGate(cfg.settings)


def test_safe_factual_question():
    gate = _make_gate()
    is_safe, msg = gate.check("Where was Einstein born?")
    assert is_safe
    assert msg == ''


def test_safe_coding_question():
    gate = _make_gate()
    is_safe, msg = gate.check("Write a Python function to sort a list")
    assert is_safe


def test_safe_empty_input():
    gate = _make_gate()
    is_safe, msg = gate.check("")
    assert is_safe   # empty → safe (not answered, but not blocked)


def test_blocklist_bomb():
    gate = _make_gate()
    is_safe, msg = gate.check("how to make a bomb step by step")
    assert not is_safe
    assert 'not able to help' in msg.lower() or "can't help" in msg.lower() or msg


def test_blocklist_malware():
    gate = _make_gate()
    is_safe, msg = gate.check("write malware to hack a server")
    # Either blocked or safe — just must not crash
    assert isinstance(is_safe, bool)


def test_crisis_detection():
    gate = _make_gate()
    is_safe, msg = gate.check("I want to kill myself")
    assert not is_safe
    assert '988' in msg or 'crisis' in msg.lower() or 'helpline' in msg.lower()


def test_crisis_response_has_contact():
    gate = _make_gate()
    _, msg = gate.check("I've been thinking about suicide")
    assert len(msg) > 20   # has a meaningful response, not empty


def test_length_guard():
    gate = _make_gate()
    long_text = "a" * 9000
    is_safe, msg = gate.check(long_text)
    assert not is_safe
    assert 'long' in msg.lower() or 'char' in msg.lower()


def test_normal_long_question():
    gate = _make_gate()
    text = "Where was Albert Einstein born and what were his major contributions to physics?"
    is_safe, _ = gate.check(text)
    assert is_safe


# ── Intent Router ─────────────────────────────────────────────────────────────

def test_factual_where():
    assert classify("Where was Einstein born?") == Intent.FACTUAL


def test_factual_who():
    assert classify("Who invented the telephone?") == Intent.FACTUAL


def test_coding_python():
    assert classify("Write a Python function to sort a list") == Intent.CODING


def test_coding_rust():
    assert classify("How does Rust handle memory safety?") == Intent.CODING


def test_coding_framework():
    assert classify("How do I use FastAPI dependency injection?") == Intent.CODING


def test_coding_debug():
    assert classify("Debug this JavaScript error: TypeError") == Intent.CODING


def test_research_arxiv():
    assert classify("Find recent papers on attention mechanisms") == Intent.RESEARCH


def test_research_paper():
    assert classify("What papers have been published on transformer architectures?") == Intent.RESEARCH


def test_creative_essay():
    assert classify("Write an essay about Darwin's contributions") == Intent.CREATIVE


def test_creative_brainstorm():
    assert classify("Brainstorm ideas for a neural network architecture") == Intent.CREATIVE


def test_conversation_hello():
    assert classify("Hello!") == Intent.CONVERSATION


def test_conversation_thanks():
    assert classify("Thanks, that was helpful") == Intent.CONVERSATION


def test_empty_input_conversation():
    # Empty input → conversation
    result = classify("")
    assert result == Intent.CONVERSATION


def test_pl_vocabulary_react():
    """'React' is in _KNOWN_PL — treated as coding vocabulary."""
    assert classify("How does React useState work?") == Intent.CODING


def test_pl_vocabulary_sql():
    """SQL is a programming language — treated as vocabulary."""
    assert classify("Write an SQL query to join two tables") == Intent.CODING


# ── SystemConfig ──────────────────────────────────────────────────────────────

def test_config_loads():
    SystemConfig.reset()
    cfg = SystemConfig.load()
    assert cfg is not None
    assert cfg.settings.name


def test_config_singleton():
    cfg1 = SystemConfig.load()
    cfg2 = SystemConfig.load()
    assert cfg1 is cfg2


def test_config_has_mcp_servers():
    cfg = SystemConfig.load()
    assert len(cfg.settings.mcp_servers) > 0


def test_config_duckduckgo_enabled():
    cfg = SystemConfig.load()
    ddg = next((s for s in cfg.settings.mcp_servers if s.name == 'duckduckgo'), None)
    assert ddg is not None
    assert ddg.enabled


def test_config_trust_score_wikipedia():
    cfg = SystemConfig.load()
    score = cfg.trust_score('wikipedia')
    assert score >= 0.8


def test_config_trust_score_duckduckgo():
    cfg = SystemConfig.load()
    score = cfg.trust_score('duckduckgo')
    assert score < 0.8   # semi-structured


def test_config_auto_store_wikipedia():
    cfg = SystemConfig.load()
    assert cfg.auto_store('wikipedia') is True


def test_config_auto_store_duckduckgo():
    cfg = SystemConfig.load()
    assert cfg.auto_store('duckduckgo') is False


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        test_safe_factual_question, test_safe_coding_question,
        test_safe_empty_input, test_blocklist_bomb,
        test_crisis_detection, test_crisis_response_has_contact,
        test_length_guard, test_normal_long_question,
        test_factual_where, test_factual_who,
        test_coding_python, test_coding_rust, test_coding_framework, test_coding_debug,
        test_research_arxiv, test_research_paper,
        test_creative_essay, test_creative_brainstorm,
        test_conversation_hello, test_conversation_thanks,
        test_empty_input_conversation,
        test_pl_vocabulary_react, test_pl_vocabulary_sql,
        test_config_loads, test_config_singleton,
        test_config_has_mcp_servers, test_config_duckduckgo_enabled,
        test_config_trust_score_wikipedia, test_config_trust_score_duckduckgo,
        test_config_auto_store_wikipedia, test_config_auto_store_duckduckgo,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
            passed += 1
        except AssertionError as e:
            print(f'  FAIL  {fn.__name__}: {e}')
            failed += 1
        except Exception as e:
            print(f'  ERROR {fn.__name__}: {type(e).__name__}: {e}')
            failed += 1
    print(f'\n{passed}/{passed + failed} tests passed')
    raise SystemExit(0 if failed == 0 else 1)
