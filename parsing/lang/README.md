# Language -> IR mappings.

`python.toml` and `cpp.toml` shipped in Phase 1. `java.toml`, `javascript.toml`,
`c.toml` and `go.toml` land in Phase 4. See plan §5 for the design and
`parsing/normalize.py` for the walker that consumes these files -- each file
documents its own schema (`[nodes]`, `[calls]`, `[binary_operator_nodes]`,
`[binary_operator_map]`, `[params]`) in its header comment.
