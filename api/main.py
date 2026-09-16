"""PolyO API -- parses code and predicts complexity. Executes nothing.

Hard invariant, enforced by tests/test_isolation.py: no module reachable
from this package may import anything under `oracle/` (which executes
untrusted code offline, during dataset labelling), and no file in this
package may contain exec/eval/compile/subprocess. Predicting from source
must never require running it.

/v1/predict lands in Phase 6, once parsing/features/models exist. This
module currently exposes only the two endpoints that infrastructure
(CI's docker build, Render's keep-alive) needs today.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="PolyO",
    description="Static, multi-language time & space complexity prediction.",
    version="0.0.0",
)


class Health(BaseModel):
    status: str


@app.get("/health", response_model=Health)
def health() -> Health:
    # Deliberately touches no model, dataset, or DB. This is the Render
    # free-tier keep-alive target (see plan SS0/SS13) and must stay instant.
    return Health(status="ok")


class Language(BaseModel):
    id: str
    tier: int


SUPPORTED_LANGUAGES: list[Language] = [
    Language(id="python", tier=1),
    Language(id="cpp", tier=1),
    Language(id="java", tier=1),
    Language(id="javascript", tier=1),
    Language(id="c", tier=2),
    Language(id="go", tier=2),
]


@app.get("/v1/languages", response_model=list[Language])
def languages() -> list[Language]:
    return SUPPORTED_LANGUAGES
