from __future__ import annotations

import re
from typing import Any

from .preflight import ALLOWED_OPERATORS, GROUP_IDENTIFIERS, RESERVED_IDENTIFIERS


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def expression_structure_key(expression: str) -> str:
    """Return an operator skeleton that abstracts fields and numeric knobs."""
    return _expression_key(expression, preserve_fields=False)


def expression_variant_key(expression: str) -> str:
    """Return a local-variant key that preserves fields but abstracts numeric knobs."""
    return _expression_key(expression, preserve_fields=True)


def _expression_key(expression: str, preserve_fields: bool) -> str:
    text = str(expression or "")
    tokens: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        number = _NUMBER_RE.match(text, index)
        if number:
            tokens.append("#")
            index = number.end()
            continue
        identifier = _IDENTIFIER_RE.match(text, index)
        if identifier:
            raw = identifier.group(0)
            name = raw.lower()
            next_nonspace = _next_nonspace(text, identifier.end())
            if next_nonspace == "(" and name in ALLOWED_OPERATORS:
                tokens.append(name)
            elif name in GROUP_IDENTIFIERS:
                tokens.append(f"group:{name}")
            elif name in RESERVED_IDENTIFIERS:
                tokens.append(name)
            else:
                tokens.append(f"field:{name}" if preserve_fields else "field")
            index = identifier.end()
            continue
        tokens.append(char)
        index += 1
    return "".join(tokens)


def _next_nonspace(text: str, index: int) -> str:
    while index < len(text) and text[index].isspace():
        index += 1
    return text[index] if index < len(text) else ""


def expression_signature_metadata(expression: str) -> dict[str, Any]:
    return {
        "structure_key": expression_structure_key(expression),
        "variant_key": expression_variant_key(expression),
    }
