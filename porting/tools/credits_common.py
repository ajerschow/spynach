"""Shared code of the credits generators (gen_spinach_authors.py, gen_easyspin_authors.py): wrapping an
acknowledgement block to the line length of the source and writing it at the end of a docstring."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

# The first words of an acknowledgement block; a block runs to the end of its docstring and is replaced as a whole.
BLOCK_STARTS = ("Spinach authors", "EasySpin authors")


def wrap_block(block: list[str], indent: int) -> list[str]:
    """Wrap the lines of a block to the line length of the source (continuation lines indented by two)."""
    out: list[str] = []
    for line in block:
        lead = len(line) - len(line.lstrip())
        out += textwrap.wrap(
            line,
            width=110 - indent,
            initial_indent="",
            subsequent_indent=" " * (lead + 2),
            break_long_words=False,
            break_on_hyphens=False,
        )
    return out


def rewrite_docstring(lines: list[str], expr: ast.Expr, block: list[str]) -> None:
    """Replace the lines of one docstring so that it ends with ``block`` (an earlier block is removed first)."""
    start, end = expr.lineno - 1, expr.end_lineno - 1
    indent = " " * expr.col_offset
    text = "\n".join(lines[start : end + 1]).rstrip()
    if not text.endswith('"""'):
        raise SystemExit(f"line {start + 1}: docstring does not end with triple double quotes")
    body = text[:-3].split("\n")
    for i, line in enumerate(body):
        if i > 0 and line.strip().startswith(BLOCK_STARTS):
            body = body[:i]
            break
    while len(body) > 1 and body[-1].strip() == "":
        body.pop()
    new = [*body, "", *[(indent + b) if b else "" for b in block], indent + '"""']
    lines[start : end + 1] = new


def apply_edits(targets: dict[Path, list[tuple[ast.Expr, list[str]]]]) -> int:
    """Write the blocks into the docstrings; returns the number of files changed. The result must stay valid Python."""
    changed = 0
    for py, found in targets.items():
        original = py.read_text(encoding="utf-8")
        lines = original.split("\n")
        for expr, block in sorted(found, key=lambda t: t[0].lineno, reverse=True):
            rewrite_docstring(lines, expr, block)
        updated = "\n".join(lines)
        ast.parse(updated)
        if updated != original:
            py.write_text(updated, encoding="utf-8")
            changed += 1
    return changed
