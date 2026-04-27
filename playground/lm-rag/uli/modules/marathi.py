"""
Marathi Language Module — implements LanguageModule protocol.
Handles Marathi text AND Marathi-English code-switching.
"""

from typing import List
from ..protocol import LanguageModule, Token, MeaningAST
from ..lexer import (
    Normalizer, tokenize, detect_language, detect_token_language,
    replace_emoji, extract_special
)
from ..semantics import tokens_to_ast
from ..writer import ast_to_text


class MarathiModule(LanguageModule):
    """Marathi language module with code-switching support."""

    def __init__(self):
        self.normalizer_mr = Normalizer(lang='mr')
        self.normalizer_en = Normalizer(lang='en')

    def detect(self, text: str) -> str:
        return detect_language(text)

    def normalize(self, text: str) -> str:
        text = replace_emoji(text)
        text, _ = extract_special(text)
        # For code-switched text: normalize Marathi parts with mr rules,
        # English parts with en rules
        words = text.split()
        result = []
        for w in words:
            lang = detect_token_language(w)
            if lang == 'en':
                # Apply English normalization
                normalized = self.normalizer_en.normalize(w)
                result.append(normalized)
            else:
                # Apply Marathi normalization
                normalized = self.normalizer_mr.normalize(w)
                result.append(normalized)
        return ' '.join(result)

    def tokenize(self, text: str) -> List[Token]:
        # spaCy English model for parsing (universal dependency labels)
        # Each token gets language tag via detect_token_language
        return tokenize(text, lang='en')  # Uses English parser for universal deps

    def to_ast(self, tokens: List[Token], text: str = '') -> MeaningAST:
        ast = tokens_to_ast(tokens, text)
        ast.source_language = 'mr'
        return ast

    def from_ast(self, ast: MeaningAST, form: str = 'statement',
                 temperature: float = 0.0) -> str:
        # For now, generate in English (Marathi generation requires
        # Marathi grammar rules for SOV word order)
        # TODO: implement Marathi word order from grammar/mr.json
        return ast_to_text(ast, lang='en', temperature=temperature)
