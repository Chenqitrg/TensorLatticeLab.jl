# Unit-square geometry from a transfer graph

`lattice_geometry(G)` gives a reference metric for understanding the graph.
It does not fit tensor eigenvalues and does not produce the physical
`TorusGeometry` envisaged for CFT measurements.

## Conventions

- Each base rank-(2,2) node is a unit square with ports
  `[left up; down right]`. Its center is a point in Z².
- A graph node view carries the original side correspondence along with its
  port permutation. Basic color names and dual flags do not supply a metric.
- The i-th output boundary port is glued to the i-th input boundary port.
  This convention uses boundary order, not index spelling or vertex names.
- Connected sides must have opposite normals. Gluing is by translation, with
  no curvature, reflection across a seam, or rotational holonomy.
- Crossing from an output to its paired input has temporal winding +1.

The result contains two integer vectors `space_period` and `time_period`.
The space period generates cycles of temporal winding zero; the time period
has winding +1. Choose the space orientation to make `det(space,time)>0`.
The time cycle is determined modulo space, so choose its representative with
`-1/2 < real(tau) <= 1/2`. Do not apply an SL(2,Z) reduction that would exchange
the distinguished spatial and temporal cycles.

For `s = space_period`, `t = time_period`, the returned quantities are

```
area          = det(s,t) = number of tensor occurrences
circumference = norm(s)
height        = area / norm(s)
relative_shift = dot(s,t) // dot(s,s)
tau           = (dot(s,t) + im * det(s,t)) / dot(s,s)
```

The relative shift is dimensionless and has an exact `Rational{Int}` value
because the periods have integer coordinates. The corresponding absolute
displacement is `relative_shift * circumference`. The graph constructors
retain their integer `shift` argument, which counts boundary-leg positions.

`positions` contains tile centers in expanded occurrence order, reduced
modulo `(s,t)` into the half-open fundamental parallelogram. The first expanded
occurrence fixes the origin. Reordering factors can therefore change absolute
positions, while the periods and geometric invariants remain the same.

The result is a `LatticeGeometry` storing the two period vectors and the tile
positions. All other quantities are computed properties. Its `area` counts
unit tiles; it is **not** the
additive area contribution to log Z. This function neither assumes nor verifies
an intended A/B checkerboard pattern across the seam.

## Reading the implementation

`src/geometry/types.jl` defines the result type and its computed properties.
`src/geometry/inference.jl` starts with the full algorithm in four stages:

1. `_square_adjacency` and `_check_square_torus`: connect the sides and check
   the elementary squares.
2. `_place_along_spanning_tree` and `_closed_walk_displacements`: place one
   copy of each vertex, then measure the mismatches on closed walks.
3. `_find_time_period` and `_find_space_period`: select cycles with one and
   zero net seam crossings, respectively.
4. `_normalize_period_basis` and `_wrap_tile_centers`: choose the documented
   representatives and return `LatticeGeometry`.

Intermediate cycle records have named fields `displacement` and
`seam_crossings`; no numerical eigenproblem is involved.

The least familiar step is `_find_time_period`. If the currently accumulated
walk crosses the seam `k` times and another walk crosses it `m` times, `gcdx`
finds integers `a,b` with `a*k + b*m = gcd(k,m)`. The same combination of their
coordinate displacements describes the combined walk. Repeating this over all
walks gives a once-crossing period if the cut is primitive. Subtracting the
appropriate number of time periods then turns every walk into a purely spatial
one. A second gcd finds the basic spatial period. Both equalities are written
next to the corresponding operations in the source.

## Why the graph determines these periods

1. Expand nested graphs with TensorGraphLab's hygienic index handling.
2. Glue internal edges and paired boundary edges. Record four neighboring
   vertices per square and the temporal winding of each edge.
3. Check that right/up translations commute. Also check that temporal winding
   sums to zero around each elementary square. These checks exclude curved
   square complexes and boundaries that are not closed spatial cuts.
4. Traverse a spanning tree to assign integer positions and winding potentials.
   Every remaining edge gives a cycle `(dx, dy, winding)`.
5. Use integer Bezout combinations to construct a cycle with winding 1.
   Subtract its temporal multiples from all cycles to find the primitive
   spatial generator. Check that the period area equals the number of tiles.

The graph must close to one connected torus. Nonprimitive temporal cuts and
incompatible side directions are rejected explicitly. All operations use
structural graph data; no TensorMaps or transfer matrices are constructed.

## Reference examples and signs

| Graph | Space period | Time period | Area | Tau |
|---|---|---|---:|---|
| `diagonal_2()` | `(1,1)` | `(-1,1)` | 2 | `im` |
| `diagonal_4()` | `(2,2)` | `(-1,1)` | 4 | `im/2` |
| `double_row_2()` | `(2,0)` | `(0,2)` | 4 | `im` |
| `shifted_row_2()` | `(2,0)` | `(1,1)` | 2 | `(1+im)/2` |
| `shifted_row_4()` | `(4,0)` | `(-1,1)` | 4 | `(-1+im)/4` |
| `tilted_4()` | `(2,2)` | `(0,2)` | 4 | `(1+im)/2` |

For rows, a positive constructor shift sends column j to output j+shift.
After output i is glued to input i, the temporal displacement is
`(-shift,layers)`, modulo the spatial period. This explains the negative real
part for `shifted_row_4()`; the sign follows the declared ports and closure,
not the positive shift used in TNRKit's reference shape labels.

The four-node tilted graph has circumference `2sqrt(2)`, height `sqrt(2)`,
and relative shift `1//2` (absolute displacement `sqrt(2)`). Its reference shape label in
TNRKit is `[sqrt(2)/2,sqrt(2),sqrt(2)/2]`. The lengths differ by a factor of
two, though their ratios give the same tau. Four unit tiles necessarily have
area four; this function does not rescale them to match that shape label.

Adjoint reverses boundary winding while retaining each base square's sides.
With the positive-orientation convention, both periods reverse and their
ratio is unchanged. This is a statement about the reference gluing, not about
complex conjugation of a numerical spectrum.
