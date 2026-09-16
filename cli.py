#!/usr/bin/env python3
"""Phase 1 CLI: file -> IR + rung-0 rule prediction (plan §14).

    python cli.py path/to/solution.py

Prints a single JSON object with the detected language, the rule-predicted
time/space classes, and a small IR summary. This is a development/demo tool,
not the served API (`api/main.py`) -- it exists to make Phase 1's pipeline
runnable end to end before there is a web frontend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from features.tabular import extract_features
from models.rule import predict_space, predict_time
from parsing.normalize import normalize_source
from parsing.parse import UnsupportedLanguageError, detect_language


def predict_file(path: Path) -> dict[str, object]:
    language = detect_language(path.name)
    source = path.read_text(encoding="utf-8")
    ir = normalize_source(source, language)
    features = extract_features(ir)
    return {
        "language": language,
        "time": predict_time(features).value,
        "space": predict_space(features).value,
        "features": {
            "max_loop_nesting_depth": features.max_loop_nesting_depth,
            "max_alloc_nesting_depth": features.max_alloc_nesting_depth,
        },
        "ir": {"nodes": len(ir.nodes), "edges": len(ir.edges)},
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: cli.py <path/to/source-file>", file=sys.stderr)
        return 2
    path = Path(argv[0])
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 1
    try:
        result = predict_file(path)
    except UnsupportedLanguageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
