"""The engine reads untrusted source on live traffic: whatever it is given it must return an
answer or raise its own typed `LoweringError`, quickly, never anything else."""

from __future__ import annotations

import time

import pytest

from analysis.engine import analyze
from analysis.nodes import LoweringError

_NL = chr(10)
_DEEP_LOOPS = (
    "def f(n):"
    + _NL
    + "".join("    " * (i + 1) + f"for i{i} in range(n):" + _NL for i in range(40))
    + "    " * 41
    + "pass"
    + _NL
)
_LONG_CHAIN = (
    "def f(n):" + _NL + "    x = 0" + _NL + "".join(f"    x = x + {i}" + _NL for i in range(2000))
)
_MANY_IFS = (
    "def f(n):"
    + _NL
    + "    x = 0"
    + _NL
    + "".join(
        f"    if n > {i}:"
        + _NL
        + f"        x += {i}"
        + _NL
        + "    else:"
        + _NL
        + f"        x -= {i}"
        + _NL
        for i in range(150)
    )
)
_CALL_CHAIN = "".join(
    f"def f{i}(n):" + _NL + f"    return f{i + 1}(n) + 1" + _NL for i in range(150)
) + ("def f150(n):" + _NL + "    return n" + _NL)
_MUTUAL = (
    "def a(n):"
    + _NL
    + "    return b(n - 1)"
    + _NL
    + _NL
    + "def b(n):"
    + _NL
    + "    return c(n - 1)"
    + _NL
    + _NL
    + "def c(n):"
    + _NL
    + "    return a(n - 1)"
    + _NL
)

_INPUTS = [
    pytest.param("", "python", id="empty"),
    pytest.param("   " + _NL * 4, "python", id="whitespace"),
    pytest.param("# only a comment" + _NL, "python", id="only-a-comment"),
    pytest.param(
        "def f(n):" + _NL + "    while True:" + _NL + "        n += 1" + _NL,
        "python",
        id="while-true",
    ),
    pytest.param(
        "def len(x):"
        + _NL
        + "    return 1"
        + _NL
        + _NL
        + "def f(xs):"
        + _NL
        + "    return len(xs)"
        + _NL,
        "python",
        id="shadowed-builtin",
    ),
    pytest.param(_DEEP_LOOPS, "python", id="forty-nested-loops"),
    pytest.param(_LONG_CHAIN, "python", id="two-thousand-statements"),
    pytest.param(_MANY_IFS, "python", id="hundred-fifty-branches"),
    pytest.param(_CALL_CHAIN, "python", id="call-chain-150"),
    pytest.param(_MUTUAL, "python", id="mutual-recursion"),
    pytest.param("int f(int n) { for (int i = 0; i < n; i++ {", "cpp", id="cpp-unbalanced"),
    pytest.param("", "cpp", id="cpp-empty"),
    pytest.param("class { }}} ;;;", "java", id="java-garbage"),
    pytest.param("func (", "go", id="go-garbage"),
    pytest.param("function ( => {", "javascript", id="js-garbage"),
    pytest.param("int main(", "c", id="c-garbage"),
]


@pytest.mark.parametrize(("source", "language"), _INPUTS)
def test_degenerate_source_is_answered_quickly(source: str, language: str) -> None:
    start = time.perf_counter()
    result = analyze(source, language)
    assert result.time.text.startswith("O(") and result.space.text.startswith("O(")
    assert time.perf_counter() - start < 5


@pytest.mark.parametrize(
    ("source", "language"),
    [
        pytest.param("this is not code at all", "python", id="prose"),
        pytest.param("def f(:", "python", id="syntax-error"),
        pytest.param("x = 1", "cobol", id="unsupported-language"),
    ],
)
def test_unparseable_input_raises_the_typed_error(source: str, language: str) -> None:
    with pytest.raises(LoweringError):
        analyze(source, language)
