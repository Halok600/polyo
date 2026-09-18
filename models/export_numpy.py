"""GNN weights -> `.npz`, and a pure-numpy reimplementation of the message
pass (plan §8: "Serving without torch... implement the 3-layer message pass
in ~50 lines of numpy. No torch and no ONNX in production.").

`models/gnn.py`'s `_GnnCore` is trained in torch; `export_gnn` dumps its
`state_dict()` straight to a `.npz` (plus small metadata: which edge kinds
and which heads are active), and `NumpyGnnModel` re-implements the identical
forward pass -- embedding lookup, N relational message-passing layers,
mean-pool readout, linear heads -- using only numpy indexing/matmul/
`np.add.at` (numpy's scatter-add). `api/` imports this module, never
`models/gnn.py` or torch.

Single-example inference only: the served API scores one request at a
time, so there is no training-time block-diagonal batching (`models/
graph_batch.py`) to reimplement here -- pooling is just a mean over one
graph's own nodes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_LAYER_NORM_EPS = 1e-5


def export_gnn(core: object, edge_kinds: tuple[str, ...], path: Path) -> None:
    """`core` is a `models.gnn._GnnCore` (typed as `object` here so this
    module never imports `models.gnn`, and so never needs torch itself)."""
    state = core.state_dict()  # type: ignore[attr-defined]
    arrays: dict[str, np.ndarray] = {
        key: tensor.detach().cpu().numpy() for key, tensor in state.items()
    }
    arrays["_edge_kinds"] = np.array(edge_kinds)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # type: ignore[arg-type]  # numpy's savez stub mistypes **kwds


class NumpyGnnModel:
    """Pure-numpy reimplementation of `_GnnCore.forward`, loaded from an
    `export_gnn` `.npz`."""

    def __init__(self, weights: dict[str, np.ndarray], edge_kinds: tuple[str, ...]):
        self.weights = weights
        self.edge_kinds = edge_kinds
        self.num_layers = len(
            {
                k
                for k in weights
                if k.startswith("encoder.layers.") and k.endswith(".self_loop.weight")
            }
        )
        head_keys = {
            k.split(".")[1] for k in weights if k.startswith("heads.") and k.endswith(".weight")
        }
        self.heads: tuple[str, ...] = tuple(sorted(head_keys))

    @classmethod
    def load(cls, path: Path) -> NumpyGnnModel:
        data = np.load(path, allow_pickle=False)
        edge_kinds = tuple(str(k) for k in data["_edge_kinds"])
        weights = {k: data[k] for k in data.files if not k.startswith("_")}
        return cls(weights, edge_kinds)

    def _layer(self, x: np.ndarray, edges_by_kind: dict[str, np.ndarray], layer: int) -> np.ndarray:
        w = self.weights
        prefix = f"encoder.layers.{layer}"
        out = x @ w[f"{prefix}.self_loop.weight"].T + w[f"{prefix}.self_loop.bias"]
        for kind in self.edge_kinds:
            pairs = edges_by_kind.get(kind)
            if pairs is None or len(pairs) == 0:
                continue
            rel_w = w[f"{prefix}.rel_linears.{kind}.weight"]
            rel_b = w[f"{prefix}.rel_linears.{kind}.bias"]
            messages = x[pairs[:, 0]] @ rel_w.T + rel_b
            np.add.at(out, pairs[:, 1], messages)
        gamma = w[f"{prefix}.norm.weight"]
        beta = w[f"{prefix}.norm.bias"]
        mean = out.mean(axis=-1, keepdims=True)
        var = out.var(axis=-1, keepdims=True)
        normed = (out - mean) / np.sqrt(var + _LAYER_NORM_EPS)
        return np.maximum(normed * gamma + beta, 0.0)

    def forward(
        self, symbol_ids: np.ndarray, edges_by_kind: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """`symbol_ids`: (num_nodes,) int64 IR symbol indices for ONE
        example (see `models/graph_batch.py`'s `SYMBOL_INDEX`). `edges_by_
        kind`: kind -> (num_edges, 2) int64 array, same shape convention as
        `ExampleGraph.edges_by_kind`. Returns raw (pre-softmax) per-class
        scores, one array per active head."""
        x = self.weights["encoder.embedding.weight"][symbol_ids]
        for layer in range(self.num_layers):
            x = self._layer(x, edges_by_kind, layer)
        pooled = x.mean(axis=0)
        return {
            head: pooled @ self.weights[f"heads.{head}.weight"].T
            + self.weights[f"heads.{head}.bias"]
            for head in self.heads
        }
