"""
English Language Module — implements LanguageModule protocol.
Pluggable: swap this for hindi.py, french.py, etc.
"""

from typing import List
from ..protocol import LanguageModule, Token, MeaningAST
from ..lexer import (
    Normalizer, tokenize, detect_language, replace_emoji, extract_special
)
from ..semantics import tokens_to_ast
from ..writer import ast_to_text


class EnglishModule(LanguageModule):
    """English language module. All rules from JSON data files."""

    def __init__(self):
        self.normalizer = Normalizer(lang='en')

    def detect(self, text: str) -> str:
        return detect_language(text)

    def normalize(self, text: str) -> str:
        text = replace_emoji(text)
        text, _ = extract_special(text)
        return self.normalizer.normalize(text)

    def tokenize(self, text: str) -> List[Token]:
        return tokenize(text, lang='en')

    def to_ast(self, tokens: List[Token], text: str = '') -> MeaningAST:
        return tokens_to_ast(tokens, text)

    def from_ast(self, ast: MeaningAST, form: str = 'statement',
                 temperature: float = 0.0) -> str:
        return ast_to_text(ast, lang='en', temperature=temperature)
