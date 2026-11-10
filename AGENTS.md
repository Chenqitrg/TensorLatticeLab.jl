# Repository instructions

- Use English for comments, docstrings, documentation and printed messages.
- Do not change the package version unless explicitly requested.
- Preserve tensor orientation and A/B colors in graph structure.
- Graph constructors return ordinary TensorGraphLab graphs, not shape-tag wrappers.
- Never materialize a transfer tensor for testing or examples. Check graph
  structure directly. Future numerical checks must apply the graph to a state
  with an explicitly bounded size; production tensors are not test fixtures.
