"""Rung 1: TF-IDF over IR symbol n-grams + logistic regression (plan §8).

Cross-language *by construction* -- it reads the IR's symbol sequence, never
raw source, so the same vectoriser/classifier pair works for any language
Phase 1+ can parse without per-language retraining. The IR alphabet is only
~40 symbols (`core/ir.py`), so even up to trigrams the vocabulary stays
small and dense compared to a natural-language TF-IDF vocabulary -- no
`max_features` cap needed.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from models.dataset import ParsedExample

_NGRAM_RANGE = (1, 3)


def symbol_sequence_text(example: ParsedExample) -> str:
    """IR symbols in emission order, space-joined -- one "token" per symbol,
    so TfidfVectorizer's n-grams over its default whitespace/word tokenizer
    are n-grams over IR symbols, not source tokens."""
    return " ".join(n.symbol for n in example.ir.nodes)


@dataclass(frozen=True, slots=True)
class TfidfModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression

    @property
    def classes(self) -> list[str]:
        return list(self.classifier.classes_)

    def predict_proba(self, examples: list[ParsedExample]) -> list[dict[str, float]]:
        x = self.vectorizer.transform([symbol_sequence_text(e) for e in examples])
        proba = self.classifier.predict_proba(x)
        return [dict(zip(self.classes, row.tolist(), strict=True)) for row in proba]

    def predict(self, examples: list[ParsedExample]) -> list[str]:
        x = self.vectorizer.transform([symbol_sequence_text(e) for e in examples])
        return list(self.classifier.predict(x))

    def decision_function(self, examples: list[ParsedExample]):
        """Raw (pre-softmax) scores -- what `models/calibrate.py` fits
        temperature scaling against, since scaling already-normalised
        probabilities from `predict_proba` would not be meaningful."""
        x = self.vectorizer.transform([symbol_sequence_text(e) for e in examples])
        return self.classifier.decision_function(x)


def fit(examples: list[ParsedExample], labels: list[str]) -> TfidfModel:
    if len(examples) != len(labels):
        raise ValueError("examples and labels must be the same length")
    texts = [symbol_sequence_text(e) for e in examples]
    vectorizer = TfidfVectorizer(ngram_range=_NGRAM_RANGE)
    x = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(x, labels)
    return TfidfModel(vectorizer=vectorizer, classifier=classifier)
