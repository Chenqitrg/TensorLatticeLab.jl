# First implementation: transfer graphs

Implement the user's request to construct the available graphs first. This review batch covers graph construction and default
coloring, before geometry fitting, center insertion and spectrum extraction.

1. Add row and diagonal graph families and named small constructors.
   Compare graph structure with independently written contraction expressions.
   Do not materialize transfer tensors: their storage grows exponentially with
   boundary width. Numerical action tests belong with the later graph-times-state
   implementation and must have explicit small size bounds.
2. Expose the four auxiliary graphs used by the 2x2 calibration as ordinary
   graphs, with orientation encoded by node views. Construct each graph directly;
   do not add a bundled default graph set or a selection-policy abstraction.
3. Add standard A/B coloring for a tensor, a pair of tensors, and the existing
   package-owned uniform/checkerboard cells directly.
   Four-tensor cells have no default mapping.
4. Provide five directly runnable examples, English documentation, and tests
   for distinct A/B labels, orientation, larger widths, nesting and invalid input.
   Coloring tests may use one-dimensional tensors without evaluating a graph.
5. Run the tests and examples, then request one independent review.

## Review focus

- Boundary order and color order must match the documented contraction.
- All named diagonal constructors preserve the graph family's color order;
  do not exchange A/B to reproduce TNRKit's special two-leg convention.
- Constructors must not truncate or coarse-grain wide networks.
- Default coloring must reject unknown labels and must preserve numerical
  TensorKit space checks, including dual directions.
- Names do not establish a numerical modular parameter or a measured area term.
  This batch must not claim to implement CFT measurement.
