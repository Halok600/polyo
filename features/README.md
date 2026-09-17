# features -- IR -> feature vectors (plan §8)

`tabular.py` ships rung 0's two features (max loop/alloc nesting depth) plus
the rung-2 family added in Phase 3 (loop/alloc counts by depth, allocation
inside a loop, recursion call count/shape, library-call counts,
branch/break/continue counts, raw node/edge counts) -- see its module
docstring for what's deliberately deferred (loop-bound shape, hash/set
lookups) and why.

`graph.py` (IR -> node/edge tensors for rung 3's GNN) lands in Phase 5.
