"""Semantics-preserving source perturbations, for invariance testing.

Each transform changes a program's TEXT but not its meaning or its asymptotic cost: padding with
dead O(1) statements, renaming local variables, adding comments, re-indenting, adding an unused
O(1) helper, and reordering top-level functions. A predictor whose answer moves under any of them
is reading the surface of the code, not its structure. (The shipped GNN's answer flipped from
O(n^2) to O(n log n) after 40 dead statements; this module exists so that can never hide again.)

Every transform returns the new source, or None when it does not apply (so callers never score a
"perturbation" that changed nothing). They are text-level and built on real parse trees (stdlib
`ast` for Python, tree-sitter for the rest), never on guessing at brace positions.
"""
from __future__ import annotations

import ast
import re
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

from parsing.parse import parse_source

_C_FAMILY = frozenset({"c", "cpp", "java", "javascript", "go"})

# Local-variable names that appear in the golden suite and are safe to rename: none is a library
# type (queue, stack, heap), a Python keyword argument (key, start, end) or a package name.
_RENAME_VOCAB = frozenset(
    {
        "nums", "target", "result", "total", "count", "left", "right", "lo", "hi", "mid", "best",
        "tmp", "out", "seen", "counts", "ordered", "grid", "rows", "cols", "visited", "dist",
        "prefix", "window", "words", "merged", "memo", "dp", "cur", "prev", "matrix", "intervals",
        "weights", "values", "capacity", "queries", "items", "order",
        "i", "j", "k", "n", "m", "r", "c", "x", "y", "a", "b", "p", "q",
    }
)  # fmt: skip

_FUNCTION_NODE_TYPES = {
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "java": {"method_declaration", "constructor_declaration"},
    "javascript": {
        "function_declaration", "function_expression", "function", "arrow_function",
        "method_definition", "generator_function_declaration",
    },
    "go": {"function_declaration", "method_declaration", "func_literal"},
}  # fmt: skip
_BLOCK_TYPES = frozenset({"block", "compound_statement", "statement_block"})


# --------------------------------------------------------------------------------------------
# literal / comment masking
# --------------------------------------------------------------------------------------------
def _protected_spans(source: str, language: str) -> list[tuple[int, int]]:
    """Character spans that renaming must never touch: string / char literals, comments, and C
    preprocessor lines."""
    spans: list[tuple[int, int]] = []
    n = len(source)
    i = 0
    line_comment = "#" if language == "python" else "//"
    while i < n:
        ch = source[i]
        if source.startswith(line_comment, i):
            j = source.find("\n", i)
            j = n if j == -1 else j
            spans.append((i, j))
            i = j
            continue
        if language != "python" and source.startswith("/*", i):
            j = source.find("*/", i + 2)
            j = n if j == -1 else j + 2
            spans.append((i, j))
            i = j
            continue
        if language in ("c", "cpp") and ch == "#":
            line_start = source.rfind("\n", 0, i) + 1
            if source[line_start:i].strip() == "":
                j = source.find("\n", i)
                j = n if j == -1 else j
                spans.append((i, j))
                i = j
                continue
        if ch in "\"'`":
            if language == "python" and source.startswith(ch * 3, i):
                j = source.find(ch * 3, i + 3)
                j = n if j == -1 else j + 3
                spans.append((i, j))
                i = j
                continue
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == ch:
                    j += 1
                    break
                if source[j] == "\n" and ch != "`":
                    break
                j += 1
            spans.append((i, j))
            i = j
            continue
        i += 1
    return spans


def _in_spans(starts: list[int], spans: list[tuple[int, int]], position: int) -> bool:
    k = bisect_right(starts, position) - 1
    return k >= 0 and spans[k][0] <= position < spans[k][1]


def _splice(source: str, byte_offset: int, text: str) -> str:
    data = source.encode("utf-8")
    return (data[:byte_offset] + text.encode("utf-8") + data[byte_offset:]).decode("utf-8")


# --------------------------------------------------------------------------------------------
# dead code
# --------------------------------------------------------------------------------------------
def _pad_statements(language: str, k: int) -> list[str]:
    lines: list[str] = []
    for idx in range(k):
        value = "0" if idx == 0 else f"pad{idx - 1} + 1"
        if language == "python":
            lines.append(f"pad{idx} = {value}")
        elif language in ("c", "cpp", "java"):
            lines.append(f"int pad{idx} = {value};")
        elif language == "javascript":
            lines.append(f"let pad{idx} = {value};")
        elif language == "go":
            lines.append(f"pad{idx} := {value}")
        else:
            raise ValueError(f"no dead-code template for {language!r}")
    return lines


def _first_function_body(source: str, language: str):  # noqa: ANN202 -- tree-sitter Node
    tree = parse_source(source, language)
    wanted = _FUNCTION_NODE_TYPES[language]
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in wanted:
            body = node.child_by_field_name("body")
            if body is not None and body.type in _BLOCK_TYPES:
                return body
        stack.extend(reversed(node.children))
    return None


def add_dead_code(source: str, language: str, k: int = 20) -> str | None:
    """Insert `k` straight-line O(1) statements at the top of the first function body."""
    if language == "python":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        func = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)),
            None,
        )
        # `ast.walk` is breadth-first; the first function in SOURCE order is what we want.
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
        if funcs:
            func = min(funcs, key=lambda n: (n.lineno, n.col_offset))
        lines = source.splitlines()
        if func is not None:
            first = func.body[0]
            indent = " " * first.col_offset
            insert_at = first.lineno - 1
        else:
            indent = ""
            insert_at = 0
        pads = [indent + stmt for stmt in _pad_statements("python", k)]
        return "\n".join([*lines[:insert_at], *pads, *lines[insert_at:]]) + "\n"
    if language not in _C_FAMILY:
        return None
    body = _first_function_body(source, language)
    if body is None:
        return None
    # The trailing newline matters: Go (and semicolon-free JS) end a statement at a line break, and
    # a one-line body like `{ return len(p) }` would otherwise run into the last pad statement.
    text = "".join("\n    " + stmt for stmt in _pad_statements(language, k)) + "\n"
    return _splice(source, body.start_byte + 1, text)


# --------------------------------------------------------------------------------------------
# renaming
# --------------------------------------------------------------------------------------------
_IDENT = re.compile(r"[A-Za-z_]\w*")


def rename_identifiers(source: str, language: str) -> str | None:
    """Rename well-known local variables (`nums` -> `nums_r`), never touching literals, comments,
    member accesses (`xs.count`), or any name used as a plain function (`def total(...)`)."""
    spans = _protected_spans(source, language)
    starts = [s for s, _ in spans]
    tokens = [m for m in _IDENT.finditer(source) if not _in_spans(starts, spans, m.start())]

    def is_member(m: re.Match[str]) -> bool:
        before = source[max(0, m.start() - 2) : m.start()]
        return before.endswith(".") or before.endswith("->") or before.endswith("::")

    called_as_function = {
        m.group()
        for m in tokens
        if not is_member(m) and source[m.end() : m.end() + 1] == "("
    }
    present = {m.group() for m in tokens}
    renameable = {
        name
        for name in _RENAME_VOCAB
        if name in present and name not in called_as_function and f"{name}_r" not in present
    }
    if not renameable:
        return None

    out: list[str] = []
    cursor = 0
    for m in tokens:
        if m.group() in renameable and not is_member(m):
            out.append(source[cursor : m.start()])
            out.append(m.group() + "_r")
            cursor = m.end()
    out.append(source[cursor:])
    result = "".join(out)
    return result if result != source else None


# --------------------------------------------------------------------------------------------
# comments and whitespace
# --------------------------------------------------------------------------------------------
def _lines_inside_multiline_spans(source: str, language: str) -> set[int]:
    inside: set[int] = set()
    for start, end in _protected_spans(source, language):
        first = source.count("\n", 0, start)
        last = source.count("\n", 0, end)
        if last > first:
            inside.update(range(first, last + 1))
    return inside


def add_comments(source: str, language: str) -> str | None:
    marker = "#" if language == "python" else "//"
    skip = _lines_inside_multiline_spans(source, language)
    out: list[str] = [f"{marker} perturbation: comment header"]
    for index, line in enumerate(source.splitlines()):
        stripped = line.strip()
        if (
            not stripped
            or index in skip
            or line.endswith("\\")
            or (language in ("c", "cpp") and stripped.startswith("#"))
            or stripped.startswith(marker)
        ):
            out.append(line)
        else:
            out.append(f"{line}  {marker} note")
    return "\n".join(out) + "\n"


def reformat_whitespace(source: str, language: str) -> str | None:
    skip = _lines_inside_multiline_spans(source, language)
    out: list[str] = []
    for index, line in enumerate(source.splitlines()):
        if index in skip:
            out.append(line)
            continue
        stripped = line.rstrip()
        if language == "python":
            indent = len(stripped) - len(stripped.lstrip(" "))
            out.append(" " * (indent * 2) + stripped.lstrip(" "))
        else:
            out.append(stripped.lstrip())
    result = "\n".join(out) + "\n"
    return result if result != source else None


# --------------------------------------------------------------------------------------------
# unused helper
# --------------------------------------------------------------------------------------------
_HELPER = {
    "python": "def unused_helper_zz(x):\n    return x + 1\n\n\n",
    "c": "int unused_helper_zz(int x) {\n    return x + 1;\n}\n\n",
    "cpp": "int unused_helper_zz(int x) {\n    return x + 1;\n}\n\n",
    "javascript": "function unused_helper_zz(x) {\n  return x + 1;\n}\n\n",
    "go": "func unused_helper_zz(x int) int {\n    return x + 1\n}\n\n",
    "java": "\n    private int unused_helper_zz(int x) {\n        return x + 1;\n    }\n",
}


def insert_unused_helper(source: str, language: str) -> str | None:
    """Add an O(1) function that nothing calls. It must not change the answer: the cost of a
    program is the cost of what is reachable, not of what is merely present."""
    if language in ("python", "c", "cpp", "javascript"):
        return _HELPER[language] + source
    tree = parse_source(source, language)
    if language == "go":
        for child in tree.root_node.children:
            if child.type in ("function_declaration", "method_declaration", "type_declaration"):
                return _splice(source, child.start_byte, _HELPER["go"])
        return None
    if language == "java":
        for child in tree.root_node.children:
            if child.type == "class_declaration":
                body = child.child_by_field_name("body")
                if body is not None:
                    return _splice(source, body.start_byte + 1, _HELPER["java"])
        return None
    return None


# --------------------------------------------------------------------------------------------
# reordering
# --------------------------------------------------------------------------------------------
def _reorder_python(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    body = tree.body
    defs = [i for i, n in enumerate(body) if isinstance(n, ast.FunctionDef)]
    if len(defs) < 2:
        return None
    # Module-level code is only safe to leave in place if it sits entirely before the first def
    # or entirely after the last (it may call the functions, which must all exist by then).
    for i, node in enumerate(body):
        if defs[0] < i < defs[-1] and not isinstance(node, ast.FunctionDef):
            return None
    lines = source.splitlines()

    def segment(node: ast.stmt) -> tuple[int, int]:
        first = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])
        assert node.end_lineno is not None
        return first - 1, node.end_lineno

    spans = [segment(body[i]) for i in defs]
    texts = ["\n".join(lines[a:b]) for a, b in spans]
    head = "\n".join(lines[: spans[0][0]])
    tail = "\n".join(lines[spans[-1][1] :])
    middle = "\n\n\n".join(reversed(texts))
    parts = [p for p in (head.rstrip("\n"), middle, tail.strip("\n")) if p]
    result = "\n\n\n".join(parts) + "\n"
    return result if result != source else None


def _reorder_slots(source: str, language: str, node_type: str) -> str | None:
    tree = parse_source(source, language)
    nodes = [c for c in tree.root_node.children if c.type == node_type]
    if len(nodes) < 2:
        return None
    data = source.encode("utf-8")
    texts = [data[n.start_byte : n.end_byte] for n in nodes]
    out = bytearray()
    cursor = 0
    for node, text in zip(nodes, reversed(texts), strict=True):
        out += data[cursor : node.start_byte] + text
        cursor = node.end_byte
    out += data[cursor:]
    result = out.decode("utf-8")
    return result if result != source else None


def reorder_functions(source: str, language: str) -> str | None:
    """Reverse the order of top-level functions, where the language does not care about order:
    Python (calls resolve at run time), JavaScript function declarations (hoisted) and Go."""
    if language == "python":
        return _reorder_python(source)
    if language == "javascript":
        return _reorder_slots(source, language, "function_declaration")
    if language == "go":
        return _reorder_slots(source, language, "function_declaration")
    return None


# --------------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Perturbation:
    name: str
    apply: Callable[[str, str], str | None]


PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation("dead_code", lambda s, lang: add_dead_code(s, lang, k=20)),
    Perturbation("dead_code_heavy", lambda s, lang: add_dead_code(s, lang, k=60)),
    Perturbation("rename", rename_identifiers),
    Perturbation("comments", add_comments),
    Perturbation("reformat", reformat_whitespace),
    Perturbation("unused_helper", insert_unused_helper),
    Perturbation("reorder_functions", reorder_functions),
)


def apply_all(source: str, language: str) -> list[tuple[str, str]]:
    """Every applicable perturbation that actually changed the text, as (name, new_source)."""
    variants: list[tuple[str, str]] = []
    for perturbation in PERTURBATIONS:
        try:
            changed = perturbation.apply(source, language)
        except Exception:  # noqa: BLE001 -- a transform that cannot handle a source is "not applicable"
            changed = None
        if changed is not None and changed != source:
            variants.append((perturbation.name, changed))
    return variants
