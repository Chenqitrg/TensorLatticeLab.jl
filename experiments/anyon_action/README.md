# Anyon insertion on a marked tube

Run `run.jl` to check the public `prepare_action` against small independent
planar contractions, including the Z3 and Fibonacci center sectors. The reference actions use alternating local gates and explicit
left or right translations at the physical seam. They cover all eight named
transfers and two six-leg geometries. The small states have at most 86 complex
coefficients. A flux-reversal control tests sensitivity to crossing phases. The
composition test stacks the open patches and retains both sets of annular
pairs. The dagger test reflects the patch and checks the inner product.

The original shortest-path experiment (`routing.jl`, `action.jl` and
`checks.jl`) chose a new seam after flattening. For two diagonal layers this
could introduce a topological twist. The public action now reads the seam
directly from the TubeGraph. The independent `canonical_seam_checks.jl`
illustrates the loss of winding information in an unmarked graph.
