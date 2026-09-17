"""Rung 3: GNN message-passing over the IR graph, shared encoder -> two
heads (plan §8).

Where rung 1 reads the IR symbol *sequence* and rung 2 reads the aggregated
tabular feature vector, this rung reads the IR *graph* itself -- nodes are
IR symbols, edges are AST_CHILD/NEXT_SIBLING/DATA_DEP/LOOP_CARRY/CALL_EDGE
(`core/ir.py`, `parsing/normalize.py`). No torch-geometric/dgl is installed
(confirmed absent), so `models/graph_batch.py`'s batching and the message
pass below are hand-written in plain torch tensor ops -- index_select,
matmul, index_add -- which is also exactly what plan §8's Phase-6 numpy
export has to reimplement, so there is no graph-library abstraction here to
translate away from later.

`fit_multitask`/`fit_single_task` are the two arms of the multi-task-vs-
single-task ablation (plan §9): a joint model shares one encoder across both
heads and can learn from an example that only has *one* dimension's label
(e.g. CodeComplex has no space_class) via a masked per-head loss, where the
single-task arm trains two fully independent models on pre-filtered,
fully-labelled data (matching rung 1/2's `with_label` convention). Either
model exposes the same `.classes` / `predict` / `predict_proba` /
`decision_function(examples)` interface as `models/gbdt.py`/`models/tfidf.py`,
so `eval/model_card.py` and `models/calibrate.py` work with rung 3 without
new glue.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn

from core.taxonomy import SpaceClass, TimeClass
from models.dataset import ParsedExample
from models.graph_batch import (
    ALL_EDGE_KINDS,
    NUM_SYMBOLS,
    ExampleGraph,
    GraphBatch,
    collate,
    to_example_graph,
)

TIME_CLASSES: tuple[str, ...] = tuple(c.value for c in TimeClass)
SPACE_CLASSES: tuple[str, ...] = tuple(c.value for c in SpaceClass)
_CLASSES_BY_DIMENSION: dict[str, tuple[str, ...]] = {"time": TIME_CLASSES, "space": SPACE_CLASSES}


class _RelationalMessagePassing(nn.Module):
    """One message-passing layer: a self-loop transform plus one linear
    transform per active edge kind, whose messages are scatter-summed into
    their destination nodes -- a small relational-GNN (R-GCN-style) layer,
    since a single shared transform can't tell "this node's parent" apart
    from "this node's loop-carried variable"."""

    def __init__(self, in_dim: int, out_dim: int, edge_kinds: tuple[str, ...]):
        super().__init__()
        self.edge_kinds = edge_kinds
        self.self_loop = nn.Linear(in_dim, out_dim)
        self.rel_linears = nn.ModuleDict({kind: nn.Linear(in_dim, out_dim) for kind in edge_kinds})
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index_by_kind: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.self_loop(x)
        for kind in self.edge_kinds:
            edge_index = edge_index_by_kind.get(kind)
            if edge_index is None or edge_index.shape[1] == 0:
                continue
            src, dst = edge_index[0], edge_index[1]
            out = out.index_add(0, dst, self.rel_linears[kind](x[src]))
        return F.relu(self.norm(out))


class _GnnEncoder(nn.Module):
    """Symbol embedding -> N relational message-passing layers -> mean-pool
    readout, one graph embedding per example."""

    def __init__(self, hidden_dim: int, num_layers: int, edge_kinds: tuple[str, ...]):
        super().__init__()
        self.embedding = nn.Embedding(NUM_SYMBOLS, hidden_dim)
        self.layers = nn.ModuleList(
            [
                _RelationalMessagePassing(hidden_dim, hidden_dim, edge_kinds)
                for _ in range(num_layers)
            ]
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = self.embedding(batch.symbol_ids)
        for layer in self.layers:
            x = layer(x, batch.edge_index)
        summed = torch.zeros(batch.num_graphs, x.shape[1], device=x.device).index_add(
            0, batch.batch_index, x
        )
        counts = (
            torch.zeros(batch.num_graphs, device=x.device)
            .index_add(0, batch.batch_index, torch.ones(batch.num_nodes, device=x.device))
            .clamp(min=1)
            .unsqueeze(1)
        )
        return summed / counts


class _GnnCore(nn.Module):
    """Shared encoder + one linear head per active dimension. Both training
    paths below use this same module; single-task just instantiates it with
    one head, so there is one message-passing implementation, not two."""

    def __init__(
        self, hidden_dim: int, num_layers: int, edge_kinds: tuple[str, ...], heads: tuple[str, ...]
    ):
        super().__init__()
        self.encoder = _GnnEncoder(hidden_dim, num_layers, edge_kinds)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, len(_CLASSES_BY_DIMENSION[name])) for name in heads}
        )

    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        embedding = self.encoder(batch)
        return {name: head(embedding) for name, head in self.heads.items()}


def _ordinal_soft_targets(true_idx: torch.Tensor, num_classes: int, tau: float) -> torch.Tensor:
    ranks = torch.arange(num_classes, device=true_idx.device).unsqueeze(0).float()
    true_ranks = true_idx.unsqueeze(1).float()
    weights = torch.exp(-(ranks - true_ranks).abs() / tau)
    return weights / weights.sum(dim=1, keepdim=True)


def ordinal_cross_entropy(
    logits: torch.Tensor, true_idx: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """Distance-weighted cross-entropy (plan §8: "ordinal loss: distance-
    weighted cross-entropy or CORAL"). The soft target distribution puts
    most mass on the true class but partial credit on ordinally-near
    classes (weight proportional to exp(-|rank_c - rank_true| / tau),
    normalised), so predicting a neighbouring class costs less loss than
    predicting a distant one -- unlike plain cross-entropy, which is blind
    to class order. A side effect worth knowing: a class with zero training
    examples can still receive some gradient signal via a neighbour's soft
    target mass, which plain one-hot cross-entropy could never provide."""
    log_probs = F.log_softmax(logits, dim=1)
    soft_targets = _ordinal_soft_targets(true_idx, logits.shape[1], tau)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def _prepare_dataset(
    examples: list[ParsedExample], labels_by_dim: dict[str, list[str | None]]
) -> tuple[list[ExampleGraph], dict[str, np.ndarray]]:
    """Converts each example's IR to an `ExampleGraph` once (not once per
    epoch), and each dimension's labels to rank-index arrays with -1
    marking "no label for this example" -- masked out of that dimension's
    loss in `_train`, not dropped from the batch, so one shared mini-batch
    can carry an example with a time label but no space label."""
    graphs = [to_example_graph(e.ir) for e in examples]
    label_indices: dict[str, np.ndarray] = {}
    for dim, labels in labels_by_dim.items():
        class_index = {c: i for i, c in enumerate(_CLASSES_BY_DIMENSION[dim])}
        label_indices[dim] = np.array(
            [class_index[label] if label is not None else -1 for label in labels], dtype=np.int64
        )
    return graphs, label_indices


def _validation_score(
    core: _GnnCore,
    val_graphs: list[ExampleGraph],
    val_labels: dict[str, np.ndarray],
    edge_kinds: tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> float:
    """Mean macro-F1 across active heads -- what early stopping in `_train`
    watches, so a joint model isn't checkpointed on a point where one head
    happens to spike while the other degrades."""
    core.eval()
    scores = []
    with torch.no_grad():
        for head in core.heads:
            y = val_labels[head]
            mask = y >= 0
            if not mask.any():
                continue
            masked_graphs = [g for g, keep in zip(val_graphs, mask, strict=True) if keep]
            preds = []
            for start in range(0, len(masked_graphs), batch_size):
                batch = collate(masked_graphs[start : start + batch_size], edge_kinds, device)
                preds.append(core(batch)[head].argmax(dim=1).cpu().numpy())
            pred_idx = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
            scores.append(f1_score(y[mask], pred_idx, average="macro", zero_division=0))
    return float(np.mean(scores)) if scores else 0.0


def _train(
    core: _GnnCore,
    train_graphs: list[ExampleGraph],
    train_labels: dict[str, np.ndarray],
    val_graphs: list[ExampleGraph] | None,
    val_labels: dict[str, np.ndarray] | None,
    *,
    edge_kinds: tuple[str, ...],
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    tau: float,
    device: torch.device,
    verbose: bool,
) -> _GnnCore:
    core.to(device)
    optimizer = torch.optim.Adam(core.parameters(), lr=lr)
    heads = tuple(core.heads.keys())
    order = list(range(len(train_graphs)))

    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        core.train()
        random.shuffle(order)
        total_loss = 0.0
        num_batches = 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = collate([train_graphs[i] for i in idx], edge_kinds, device)
            outputs = core(batch)
            loss = torch.tensor(0.0, device=device)
            for head in heads:
                y = train_labels[head][idx]
                mask = y >= 0
                if not mask.any():
                    continue
                mask_t = torch.from_numpy(mask).to(device)
                logits = outputs[head][mask_t]
                y_t = torch.from_numpy(y[mask]).to(device)
                loss = loss + ordinal_cross_entropy(logits, y_t, tau)
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                num_batches += 1

        if val_graphs is not None and val_labels is not None:
            score = _validation_score(core, val_graphs, val_labels, edge_kinds, batch_size, device)
            if verbose:
                avg_loss = total_loss / max(num_batches, 1)
                print(f"epoch {epoch}: train_loss={avg_loss:.4f} val_macro_f1={score:.4f}")
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(
                            f"early stopping at epoch {epoch} "
                            f"(best val_macro_f1={best_score:.4f})"
                        )
                    break
        elif verbose:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"epoch {epoch}: train_loss={avg_loss:.4f}")

    if best_state is not None:
        core.load_state_dict(best_state)
    return core


def _default_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _softmax_np(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class GnnModel:
    """One dimension's view over a (possibly shared) `_GnnCore` -- same
    `.classes` / `predict` / `predict_proba` / `decision_function(examples)`
    interface as `models/gbdt.py`/`models/tfidf.py`. Unlike those rungs,
    `.classes` is always the *full* taxonomy (`core.taxonomy`), not just the
    labels seen during training -- the ordinal loss needs the complete rank
    range to be meaningful even for a class this model never saw."""

    core: _GnnCore
    dimension: str
    edge_kinds: tuple[str, ...]
    device: torch.device
    batch_size: int = 256

    @property
    def classes(self) -> list[str]:
        return list(_CLASSES_BY_DIMENSION[self.dimension])

    def decision_function(self, examples: list[ParsedExample]) -> np.ndarray:
        """Raw per-class scores (pre-softmax) -- what `models/calibrate.py`
        fits temperature scaling against, matching rung 1/2's contract."""
        self.core.eval()
        graphs = [to_example_graph(e.ir) for e in examples]
        if not graphs:
            return np.zeros((0, len(self.classes)))
        chunks = []
        with torch.no_grad():
            for start in range(0, len(graphs), self.batch_size):
                chunk = graphs[start : start + self.batch_size]
                batch = collate(chunk, self.edge_kinds, self.device)
                logits = self.core(batch)[self.dimension]
                chunks.append(logits.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def predict_proba(self, examples: list[ParsedExample]) -> list[dict[str, float]]:
        proba = _softmax_np(self.decision_function(examples))
        return [dict(zip(self.classes, row.tolist(), strict=True)) for row in proba]

    def predict(self, examples: list[ParsedExample]) -> list[str]:
        scores = self.decision_function(examples)
        classes = self.classes
        return [classes[i] for i in scores.argmax(axis=1)]


def fit_multitask(
    train_examples: list[ParsedExample],
    train_time_labels: list[str | None],
    train_space_labels: list[str | None],
    *,
    val_examples: list[ParsedExample] | None = None,
    val_time_labels: list[str | None] | None = None,
    val_space_labels: list[str | None] | None = None,
    edge_kinds: tuple[str, ...] = ALL_EDGE_KINDS,
    hidden_dim: int = 64,
    num_layers: int = 3,
    batch_size: int = 256,
    max_epochs: int = 20,
    patience: int = 3,
    lr: float = 1e-3,
    tau: float = 1.0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[GnnModel, GnnModel]:
    """Trains one shared encoder with both heads jointly -- the "multi-task"
    arm of the multi-task-vs-single-task ablation (plan §9). Returns
    `(time_model, space_model)`, two `GnnModel` views over the same
    underlying core."""
    resolved_device = device or _default_device()
    train_graphs, train_labels = _prepare_dataset(
        train_examples, {"time": train_time_labels, "space": train_space_labels}
    )
    val_graphs: list[ExampleGraph] | None = None
    val_labels: dict[str, np.ndarray] | None = None
    if val_examples is not None:
        val_graphs, val_labels = _prepare_dataset(
            val_examples, {"time": val_time_labels or [], "space": val_space_labels or []}
        )

    core = _GnnCore(hidden_dim, num_layers, edge_kinds, heads=("time", "space"))
    core = _train(
        core,
        train_graphs,
        train_labels,
        val_graphs,
        val_labels,
        edge_kinds=edge_kinds,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        tau=tau,
        device=resolved_device,
        verbose=verbose,
    )
    core.eval()
    time_model = GnnModel(
        core=core,
        dimension="time",
        edge_kinds=edge_kinds,
        device=resolved_device,
        batch_size=batch_size,
    )
    space_model = GnnModel(
        core=core,
        dimension="space",
        edge_kinds=edge_kinds,
        device=resolved_device,
        batch_size=batch_size,
    )
    return time_model, space_model


def fit_single_task(
    train_examples: list[ParsedExample],
    train_labels: list[str],
    dimension: str,
    *,
    val_examples: list[ParsedExample] | None = None,
    val_labels: list[str] | None = None,
    edge_kinds: tuple[str, ...] = ALL_EDGE_KINDS,
    hidden_dim: int = 64,
    num_layers: int = 3,
    batch_size: int = 256,
    max_epochs: int = 20,
    patience: int = 3,
    lr: float = 1e-3,
    tau: float = 1.0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> GnnModel:
    """Trains an independent single-head encoder for one dimension only --
    the "single-task" arm of the same ablation. Every example here already
    has a label for this dimension (callers pass pre-filtered
    examples/labels, matching rung 1/2's `with_label` convention), since
    there is no second head left for an unlabelled example to still help."""
    if len(train_examples) != len(train_labels):
        raise ValueError("train_examples and train_labels must be the same length")
    if dimension not in _CLASSES_BY_DIMENSION:
        raise ValueError(f"unknown dimension: {dimension!r}")
    resolved_device = device or _default_device()
    train_graphs, train_label_map = _prepare_dataset(
        train_examples, {dimension: list(train_labels)}
    )

    val_graphs: list[ExampleGraph] | None = None
    val_label_map: dict[str, np.ndarray] | None = None
    if val_examples is not None and val_labels is not None:
        val_graphs, val_label_map = _prepare_dataset(val_examples, {dimension: list(val_labels)})

    core = _GnnCore(hidden_dim, num_layers, edge_kinds, heads=(dimension,))
    core = _train(
        core,
        train_graphs,
        train_label_map,
        val_graphs,
        val_label_map,
        edge_kinds=edge_kinds,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        tau=tau,
        device=resolved_device,
        verbose=verbose,
    )
    core.eval()
    return GnnModel(
        core=core,
        dimension=dimension,
        edge_kinds=edge_kinds,
        device=resolved_device,
        batch_size=batch_size,
    )
