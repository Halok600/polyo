"""How trustworthy are the corpus labels the ML arm trains on? An independent audit.

    python -m eval.label_audit --out eval/LABEL_AUDIT.md

`tests/data/corpus_audit_v1.json` holds, for programs of the BigO(Bench) / CodeComplex test splits:
the label the corpus gives, what the symbolic engine answered at first contact, and what an
independent agent decided from the code alone under the project's conventions (worst case, input
quantities as sizes). Two samples:

  * random        a deterministic random sample: it estimates how often each party is right;
  * disagreement  programs the engine called "certain" and the corpus label contradicted: it shows
                  which side the independent reading sides with.

Agents' "low confidence" answers are left out; the file keeps them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "tests" / "data" / "corpus_audit_v1.json"
DIMENSIONS = ("time", "space")


def usable(entries: Sequence[dict[str, object]], kind: str) -> list[dict[str, object]]:
    return [e for e in entries if e["kind"] == kind and e["confidence"] != "low"]


def random_sample_table(entries: Sequence[dict[str, object]], dimension: str) -> dict[str, int]:
    """Counts of who agrees with the independent reading, on rows where the corpus has a label."""
    out: Counter[str] = Counter()
    for e in usable(entries, "random"):
        corpus, engine, agent = (
            e[f"corpus_{dimension}"],
            e[f"engine_{dimension}"],
            e[f"agent_{dimension}"],
        )
        if corpus is None:
            continue
        out["rows"] += 1
        out["corpus right"] += corpus == agent
        out["engine right"] += engine == agent
        out["both right"] += corpus == agent == engine
        out["neither right"] += corpus != agent and engine != agent
    return dict(out)


def disagreement_table(entries: Sequence[dict[str, object]], dimension: str) -> dict[str, int]:
    out: Counter[str] = Counter()
    for e in usable(entries, "disagreement"):
        corpus, engine, agent = (
            e[f"corpus_{dimension}"],
            e[f"engine_{dimension}"],
            e[f"agent_{dimension}"],
        )
        if corpus is None:
            continue
        out["rows"] += 1
        if agent == engine:
            out["engine right (the corpus label is wrong, or follows another convention)"] += 1
        elif agent == corpus:
            out["corpus right (the engine is wrong)"] += 1
        else:
            out["neither"] += 1
    return dict(out)


def by_source(entries: Sequence[dict[str, object]], dimension: str) -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = {}
    for e in usable(entries, "random"):
        if e[f"corpus_{dimension}"] is None:
            continue
        cell = out.setdefault(str(e["source"]), [0, 0])
        cell[1] += 1
        cell[0] += e[f"corpus_{dimension}"] == e[f"agent_{dimension}"]
    return {k: (v[0], v[1]) for k, v in out.items()}


def _share(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.0%})" if total else "n/a"


def render(entries: Sequence[dict[str, object]]) -> str:
    lines = [
        "# Label audit: how right is the corpus the model trains on?",
        "",
        "For programs of the BigO(Bench) and CodeComplex test splits: the label the corpus gives,",
        "what the symbolic engine answered at first contact, and what an independent agent decided",
        "from the code alone under the project's conventions (worst case, the quantities read as",
        "sizes). The random sample estimates how often each party is right; the disagreement",
        "sample",
        "(the engine was 'certain' and the corpus label contradicted it) shows which side the",
        "independent reading takes. Agents' low-confidence answers are left out.",
        "",
        "## Random sample: who agrees with an independent reading of the code",
        "",
        "| dimension | rows | corpus label | engine (first contact) | both | neither |",
        "|---|---|---|---|---|---|",
    ]
    for dimension in DIMENSIONS:
        t = random_sample_table(entries, dimension)
        n = t.get("rows", 0)
        lines.append(
            f"| {dimension} | {n} | {_share(t.get('corpus right', 0), n)} | "
            f"{_share(t.get('engine right', 0), n)} | {_share(t.get('both right', 0), n)} | "
            f"{_share(t.get('neither right', 0), n)} |"
        )
    lines += ["", "Corpus label right, by source (random sample):", ""]
    for dimension in DIMENSIONS:
        cells = ", ".join(
            f"{source} {_share(*cell)}"
            for source, cell in sorted(by_source(entries, dimension).items())
        )
        lines.append(f"* {dimension}: {cells}")
    lines += [
        "",
        "## Disagreements: the engine was 'certain' and the corpus label said otherwise",
        "",
    ]
    for dimension in DIMENSIONS:
        t = disagreement_table(entries, dimension)
        n = t.pop("rows", 0)
        lines.append(f"**{dimension}** ({n} programs):")
        lines += [f"* {_share(count, n)} {what}" for what, count in sorted(t.items())]
        lines.append("")
    lines += [
        "## Reading the numbers",
        "",
        "BigO(Bench) labels a program by measuring it against the dimension of its own test",
        "harness: a loop over a small, fixed-size input reads as O(1) there, and anything that",
        "does not scale with that harness dimension does too. The project's convention is the",
        "worst case as a function of the quantities the program reads. The two disagree most on",
        "exactly the programs that dominate the corpus (competitive-programming solutions), so a",
        "model trained on those labels learns the measurement convention, not the one users",
        "expect, and it is capped by the label noise measured above whatever its architecture.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    entries = json.loads(AUDIT.read_text(encoding="utf-8"))
    report = render(entries)
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
