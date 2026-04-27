"""
ULI Writer — Reverse pipeline: MeaningAST → text.
Template selection + slot filling + grammar rules. No model.
"""

import json
import os
import random
from typing import List, Optional
from .protocol import MeaningAST, Entity


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ============================================================
# Templates — how to express different AST types
# ============================================================

# Factual answer templates (ordered by verbosity)
ANSWER_TEMPLATES = {
    'factual_short': [
        "{answer}",
    ],
    'factual_sentence': [
        "{answer} is the {target} of {context}.",
        "The {target} is {answer}.",
        "{answer}.",
    ],
    'factual_full': [
        "The {target} of {context} is {answer}.",
        "{answer} is the {target} of {context}.",
        "Based on the available information, {answer}.",
    ],
    'explanation': [
        "{reason}.",
        "This is because {reason}.",
        "The reason is that {reason}.",
    ],
    'comparison': [
        "{entity_a} is {comparison} than {entity_b}.",
        "Comparing {entity_a} and {entity_b}: {comparison}.",
    ],
    'unanswerable': [
        "I couldn't find a confident answer for this.",
        "This question cannot be answered with the available information.",
    ],
    'disambiguation': [
        "This could refer to: {options}.",
        "The term '{term}' has multiple meanings: {options}.",
    ],
}


def ast_to_text(ast: MeaningAST, lang: str = 'en',
                temperature: float = 0.0) -> str:
    """Convert MeaningAST to text string.

    temperature=0.0: deterministic (always first template, base words)
    temperature=0.5: moderate variation (random template, some synonyms)
    temperature=1.0: maximum variation (rare templates, rare synonyms)
    """

    if not ast.predicate and not ast.entities and not ast.agent:
        # Minimal AST — just return whatever text we have
        if ast.source:
            return ast.source
        return ''

    # Choose template based on AST type and intent
    templates = _select_templates(ast)

    # Pick template (temperature controls randomness)
    if temperature <= 0.0 or len(templates) <= 1:
        template = templates[0]
    else:
        # Weighted random: lower indices are more likely at low temperature
        weights = [1.0 / (i + 1) ** (1.0 / max(temperature, 0.01))
                   for i in range(len(templates))]
        template = random.choices(templates, weights=weights, k=1)[0]

    # Fill template slots
    text = _fill_template(template, ast)

    return text.strip()


def _select_templates(ast: MeaningAST) -> List[str]:
    """Select appropriate templates based on AST content."""

    # If we have a predicate + agent + patient, build from structure
    if ast.predicate and ast.agent and ast.agent.text != '?':
        return [_build_from_ast(ast)]

    # If we have a direct answer entity, use factual templates
    if ast.intent == 'factual' or (ast.patient and ast.patient.text != '?'):
        return ANSWER_TEMPLATES.get('factual_sentence', [''])

    if ast.intent == 'explanation' and ast.reason:
        return ANSWER_TEMPLATES.get('explanation', [''])

    if ast.intent == 'comparison':
        return ANSWER_TEMPLATES.get('comparison', [''])

    if ast.intent == 'unanswerable':
        return ANSWER_TEMPLATES.get('unanswerable', [''])

    # Default: construct from AST structure
    return [_build_from_ast(ast)]


def _build_from_ast(ast: MeaningAST) -> str:
    """Build sentence directly from AST structure (SVO for English)."""
    parts = []

    # Agent
    if ast.agent and ast.agent.text and ast.agent.text != '?':
        parts.append(ast.agent.text)

    # Predicate (verb)
    if ast.predicate:
        verb = ast.predicate
        # Basic tense inflection
        if ast.tense == 'past' and not verb.endswith('ed'):
            verb = verb + 'ed'  # Simplified — irregular forms come from vocab DB
        if ast.negation:
            verb = 'did not ' + ast.predicate if ast.tense == 'past' else 'does not ' + ast.predicate
        parts.append(verb)

    # Patient/theme
    if ast.patient and ast.patient.text and ast.patient.text != '?':
        parts.append(ast.patient.text)
    elif ast.theme and ast.theme.text:
        parts.append(ast.theme.text)

    # Location
    if ast.location and ast.location.text:
        parts.append(f"in {ast.location.text}")

    # Time
    if ast.time and ast.time.text:
        parts.append(f"in {ast.time.text}")

    # Manner
    if ast.manner:
        parts.append(ast.manner)

    result = ' '.join(parts)
    if result and not result.endswith('.'):
        result += '.'

    return result


def _fill_template(template: str, ast: MeaningAST) -> str:
    """Fill template slots from AST fields."""
    replacements = {
        '{answer}': _get_answer_text(ast),
        '{target}': ast.question_target or 'answer',
        '{context}': _get_context_text(ast),
        '{reason}': ast.reason or '',
        '{entity_a}': '',
        '{entity_b}': '',
        '{comparison}': '',
        '{options}': ', '.join(ast.entities) if ast.entities else '',
        '{term}': ast.entities[0] if ast.entities else '',
    }

    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)

    return result


def _get_answer_text(ast: MeaningAST) -> str:
    """Extract the primary answer from the AST."""
    # Check common answer locations
    if ast.agent and ast.agent.text and ast.agent.text != '?' and ast.question_target == 'agent':
        return ast.agent.text
    if ast.patient and ast.patient.text and ast.patient.text != '?':
        return ast.patient.text
    if ast.theme and ast.theme.text:
        return ast.theme.text
    if ast.location and ast.location.text and ast.question_target == 'location':
        return ast.location.text
    if ast.time and ast.time.text and ast.question_target == 'time':
        return ast.time.text
    if ast.entities:
        return ast.entities[0]
    return ast.predicate or ''


def _get_context_text(ast: MeaningAST) -> str:
    """Extract context description from AST."""
    parts = []
    if ast.location and ast.location.text:
        parts.append(ast.location.text)
    if ast.time and ast.time.text:
        parts.append(ast.time.text)
    if not parts and ast.entities:
        parts = ast.entities[:2]
    return ', '.join(parts) if parts else 'the query'
