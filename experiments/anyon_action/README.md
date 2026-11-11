# Investigating the canonical n|1 seam

This is an exploratory implementation, not a public package API. Loading
TensorLatticeLab does not load these files or expose `prepare_action`.

The prototype caps both ordered boundaries of a ribbon graph, checks that the
result is a connected sphere, and finds a shortest path through its faces.
Each crossed edge receives an inverse half-braiding. The resulting planar
contraction is compiled once, with the state included throughout contraction.
All numerical fixtures are small; no full transfer tensor is materialized.

## Observed failure

For the four-leg diagonal transfer G, the generated action agrees with an
independently written planar action in all nine Z3-double sectors and all four
Fibonacci-double sectors. Its adjoint also satisfies the inner-product check.
The ordinary-space checks and ten graph-family checks pass.

However, flattening a composition G composed with G and choosing a new shortest
cut does not preserve composition. In Z3-double sector (charge, flux), the
observed relation is

    action(flatten(G composed with G), a) = twist(a) * action(G, a)^2

The residual after removing this phase is about 5e-16. Four Z3-double sectors
and two Fibonacci-double sectors fail the uncorrected composition check.
Trivial-sector tests alone would miss the problem.

The shortest-cut rule is rejected. The intended public convention fixes the
input and output gaps between legs n and 1 and connects them directly in an
ordered strip, without additional winding. Users should not supply a cut or
choose another convention. Composition must preserve this seam.
No numerical phase correction is applied to hide the shortest-path failure.

## Connectivity versus an ordered strip

`canonical_seam_checks.jl` tests a separate structural issue using only a
two-leg state with three-dimensional local spaces. Two cyclic translations
composed together have the same TensorGraph connectivity as two layers of
straight identity maps: their ordered ports, colors and external boundaries
agree up to relabeling and reordering of occurrences. TensorGraph equality
also deliberately ignores hierarchical factoring.

Nevertheless, applying two charge-aware left rotations through the n|1 seam
gives the inverse topological twist, whereas the straight identity layers act
as the identity. This diagnostic does not use the shortest-path prototype.

Thus an implementation must distinguish connectivity from a chosen strip lift.
An ordered layer presentation can retain such information even when its flattened
connectivity cannot. Fixing the seam convention does not by itself make an
unmarked connectivity graph a complete representation of an arbitrary strip
diagram. This does not rule out automatic recovery for a more restricted
lattice graph family; it does rule out silently treating all equal flattened
graphs as the same compositional tube operator.

## Reproduce

Run `run.jl` with this workspace's TensorGraphLab checkout, whose internal graph
compiler accepts an explicit contraction tree. The regression intentionally
reports six failed composition checks. This script is separate from the package
test suite and is not a runnable success example.

Run `canonical_seam_checks.jl` independently for the small, passing regression
that demonstrates the connectivity/strip-lift distinction.
