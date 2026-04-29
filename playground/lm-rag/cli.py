#!/usr/bin/env python3
"""
ULI reasoning engine — command-line interface.

Interactive REPL:
    python3 cli.py

Single question:
    python3 cli.py --ask "Where was Einstein born?"

Pipe mode (one question per line):
    echo "What is photosynthesis?" | python3 cli.py --pipe

Options:
    --ask TEXT     Answer a single question and exit
    --pipe         Read questions from stdin, one per line
    --lang LANG    Language code (default: auto-detect)
    --no-web       Disable web search (KG-only)
    --verbose      Show reasoning steps (intent, source, hops)
    --version      Print version info and exit
"""

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Rich TUI helpers ─────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def _make_console():
    if _HAS_RICH:
        return Console(stderr=True)
    return None


def _step(console, icon, label, detail='', style='dim'):
    """Print a single thinking step."""
    if not console:
        if detail:
            print(f'  [{label}] {detail}', file=sys.stderr)
        return
    line = Text()
    line.append(f'  {icon} ', style='bold')
    line.append(label, style='bold')
    if detail:
        line.append(f'  {detail}', style=style)
    console.print(line)


def _answer_panel(console, answer, source=''):
    """Print the final answer."""
    if not console:
        print(f'\n{answer}')
        return
    # Source badge
    badge = ''
    if source == 'web':
        badge = ' [dim](from web)[/dim]'
    elif source == 'kg':
        badge = ' [dim](from knowledge graph)[/dim]'
    elif source == 'personal_kb':
        badge = ' [dim](from memory)[/dim]'

    console.print()
    console.print(Panel(
        answer,
        title=f'[bold]answer[/bold]{badge}',
        title_align='left',
        border_style='green',
        padding=(0, 1),
    ), highlight=False)


# ── Engine ────────────────────────────────────────────────────────────────────

def _build_engine(no_web: bool = False, console=None):
    """Initialise safety gate + reasoner + dialogue engine."""
    _step(console, ':', 'loading', 'config + safety gate + knowledge graph')

    from uli.system_prompt import SafetyGate, SystemConfig
    from uli.router import classify
    from uli.reasoner import GraphReasoner
    from uli.mcp_client import MCPClient
    from uli.dialogue import DialogueEngine

    cfg = SystemConfig.load()
    gate = SafetyGate(cfg.settings)
    reasoner = GraphReasoner()

    mcp = None
    n_servers = 0
    if not no_web:
        servers = cfg.settings.mcp_servers if hasattr(cfg.settings, 'mcp_servers') else []
        if servers:
            mcp = MCPClient(servers)
            reasoner._mcp = mcp
            n_servers = len([s for s in servers if getattr(s, 'enabled', True)])

    dialogue = DialogueEngine(reasoner, cfg.settings)

    if no_web:
        _step(console, '>', 'ready', 'knowledge graph only (web disabled)')
    else:
        _step(console, '>', 'ready', f'{n_servers} search providers connected')

    return gate, reasoner, classify, dialogue


def _answer(question: str, gate, reasoner, classify_fn, dialogue,
            lang: str = None, verbose: bool = False, console=None):
    """Run one question through the full pipeline, showing thinking steps.
    Returns (answer_text, source_tag)."""
    t0 = time.monotonic()

    # Step 0: Safety gate
    _step(console, '?', 'safety', 'checking input')
    safe, reason = gate.check(question)
    if not safe:
        _step(console, '!', 'blocked', reason, style='red')
        return reason, 'blocked'

    # Step 1: Intent classification
    intent = classify_fn(question)
    _step(console, '*', 'intent', intent.value)

    # Step 2: Dialogue pipeline (ATTENTION -> RULES -> KNOWLEDGE -> TEMPLATE -> PRAGMATICS)
    _step(console, '~', 'thinking', 'attention + rules + knowledge lookup')
    response = dialogue.respond(question, lang=lang or 'en')

    # Show what source answered
    source_labels = {
        'kg': 'knowledge graph',
        'web': 'web search',
        'personal_kb': 'memory',
        'unknown': 'no match found',
    }
    source_label = source_labels.get(response.source, response.source)
    icon = '>' if response.source != 'unknown' else '!'
    _step(console, icon, 'source', source_label)

    if response.slot:
        _step(console, '-', 'slot', response.slot, style='dim')

    elapsed = time.monotonic() - t0
    _step(console, '.', 'done', f'{elapsed:.1f}s', style='dim')

    answer = _trim_answer(response.answer)
    if response.follow_up:
        answer = f'{answer}\n{response.follow_up}'

    return answer, response.source


def _trim_answer(text: str, max_chars: int = 500) -> str:
    """For long web snippets, extract first complete sentence(s) up to max_chars."""
    if len(text) <= max_chars:
        return text
    import re
    truncated = text[:max_chars]
    match = list(re.finditer(r'[.!?]\s', truncated))
    if match:
        cut = match[-1].end()
        return text[:cut].rstrip()
    return truncated.rstrip() + '...'


# ── REPL ──────────────────────────────────────────────────────────────────────

def _repl(gate, reasoner, classify_fn, dialogue, lang: str, verbose: bool,
          console=None):
    """Interactive read-eval-print loop."""
    try:
        import readline
    except ImportError:
        pass

    out = Console() if _HAS_RICH else None

    if out:
        out.print()
        out.print(Panel(
            '[bold]ULI[/bold] reasoning engine\n'
            '[dim]type a question, "exit" to quit[/dim]',
            border_style='blue',
            padding=(0, 1),
        ))
    else:
        print('ULI reasoning engine  (type "exit" or Ctrl-D to quit)')
        print('─' * 50)

    while True:
        try:
            if out:
                out.print()
            question = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ('exit', 'quit', 'q', ':q', 'bye'):
            break

        answer, source = _answer(question, gate, reasoner, classify_fn,
                                dialogue, lang=lang, verbose=verbose,
                                console=console)

        _answer_panel(out or console, answer, source)


def _pipe_mode(gate, reasoner, classify_fn, dialogue, lang: str, verbose: bool):
    """Read questions from stdin, write answers to stdout."""
    for line in sys.stdin:
        question = line.strip()
        if not question:
            continue
        answer, _ = _answer(question, gate, reasoner, classify_fn, dialogue,
                           lang=lang, verbose=verbose)
        print(answer, flush=True)


def main():
    parser = argparse.ArgumentParser(
        prog='uli',
        description='ULI deep reasoning engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--ask', metavar='QUESTION',
                        help='Answer a single question and exit')
    parser.add_argument('--pipe', action='store_true',
                        help='Read questions from stdin, one per line')
    parser.add_argument('--lang', default=None,
                        help='Language code (default: auto-detect)')
    parser.add_argument('--no-web', action='store_true',
                        help='Disable web search — use knowledge graph only')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show intent classification and answer source')
    parser.add_argument('--version', action='store_true',
                        help='Print version and exit')
    parser.add_argument('--no-tui', action='store_true',
                        help='Disable rich TUI (plain text output)')
    args = parser.parse_args()

    if args.version:
        print('ULI reasoning engine')
        print('  KG: SQLite + graph inference')
        print('  Search: SearXNG + DDG + Wikipedia + Tavily + 12 more')
        print('  Similarity: grammar+WordNet (STS-B 0.7358, no neural model)')
        print('  API: OpenAI-compatible  (python3 -m api.server)')
        print('  MCP: stdio/SSE  (python3 -m api.mcp_server)')
        return

    console = None if args.no_tui or args.pipe else _make_console()

    try:
        gate, reasoner, classify_fn, dialogue = _build_engine(
            no_web=args.no_web, console=console)
    except Exception as e:
        print(f"Failed to initialise: {e}", file=sys.stderr)
        sys.exit(1)

    if args.ask:
        answer, source = _answer(args.ask, gate, reasoner, classify_fn,
                                 dialogue, lang=args.lang, verbose=args.verbose,
                                 console=console)
        out = Console() if _HAS_RICH and not args.no_tui else None
        _answer_panel(out, answer, source)

    elif args.pipe:
        _pipe_mode(gate, reasoner, classify_fn, dialogue, args.lang, args.verbose)

    else:
        _repl(gate, reasoner, classify_fn, dialogue, args.lang, args.verbose,
              console=console)


if __name__ == '__main__':
    main()
