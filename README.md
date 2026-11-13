# TensorLatticeLab

Periodic 2D tensor cells, transfer graphs and reference lattice geometry, built with
[TensorGraphLab](https://github.com/Chenqitrg/TensorGraphLab.jl).

**Current scope:** tensor-cell types and constructors, local leg operations,
CDL tensors, graph constructors, sector-resolved transfer actions, unit-square lattice
geometry, A/B coloring and sector-resolved transfer diagonalization.
Model libraries, physical geometry fitting and
`CFTSpectrum` are not implemented yet. There is no `measure` entry point in this
first review batch.

ExperimentLab provides the algorithm configuration interface for diagonalization.
Cells can represent UV tensors or renormalized tensors; no fixed-point
assumption is imposed. LoopTNRLab depends on this package for its input cells.

## Source organization

- `src/cells/`: periodic cell types, constructors, bond operations, CDL tensors
  and display.
- `src/transfers/`: transfer graph constructors, tensor coloring and static operators.
- `src/geometry/`: lattice geometry types and inference from graph connections.

- `src/ED/`: transfer-matrix algorithm configurations and matrix-free Arnoldi diagonalization.

## Construct a tensor cell

```julia
using TensorKit, TensorLatticeLab

# A single-entry example; no transfer tensor is constructed.
gate = ones(ComplexF64, ℂ^1 ⊗ ℂ^1 ← ℂ^1 ⊗ ℂ^1)
uniform = UniformGates(gate)
checkerboard = CheckerboardGates(gate, 2gate)
plaquette = PlaquetteGates(checkerboard)
TensorLatticeLab.assertconsistency(plaquette)
copy_of_cell = TensorLatticeLab.standardize(plaquette)
```

`AbstractTensorCell`, its three concrete types, random cell constructors,
`mul_west/south/north/east`, `cdl_tensor` and `random_cdl_tensor` are defined here.
The cell convention remains `(West, South; North, East)`, with rows increasing
southward. Periodic bonds are checked and standardized by cell construction.
`standardize` and `assertconsistency` are qualified interfaces owned by this
package. PMPSLab's corresponding functions continue to operate on MPS objects.

To run LoopTNR, import both `TensorLatticeLab` and `LoopTNRLab`. The latter
provides the algorithm and the conversion to `PeriodicMPS`, but does not
re-export the cell API.

## Inspect a transfer graph

```julia
using TensorKit, TensorGraphLab, TensorLatticeLab

G = diagonal_4()
showgraph(G; name=:transfer)
```

This is the same graph as:

```julia
A = TensorNode(:A, (2, 2))
B = TensorNode(:B, (2, 2))

@tube annular=(i1, r) transfer[o1 o2 o3 o4; p1 p2 p3 p4] :=
    A[o2 o3; i2 i3] * A[o4 r; i4 i1] *
    B[i1 i2; p1 p2] * B[i3 i4; p3 p4] * id[o1; r]

@assert G == transfer
```

The nodes retain formal leg orientation and color labels, with no tensor data,
dimensions or sectors. Each transfer constructor returns a `TubeGraph`. Inspect its open planar representative with `open_graph(G)`. For ordinary graph composition, `TensorGraphLab.closed_graph(G)` glues the annular pairs and removes identity wires.

## Available graphs

| Constructor | Reference geometry `[h, L, shift]` | Boundary legs per side |
|---|---|---:|
| `diagonal_2()` | `[√2, √2, 0]` | 2 |
| `shifted_row_2()` | `[1, 2, 1]` | 2 |
| `double_row_2()` | `[2, 2, 0]` | 2 |
| `diagonal_4()` | `[√2, 2√2, 0]` | 4 |
| `row_4()` | `[1, 4, 0]` | 4 |
| `shifted_row_4()` | `[1, 4, 1]` | 4 |
| `shifted_row_8()` | `[1, 8, 1]` | 8 |
| `TensorLatticeLab.tilted_4()` | `[√2/2, √2, √2/2]` | 4 |

The triplets describe the reference lattice drawings, **not** the numerical
modular parameter of an anisotropic tensor network. They are documentation,
not inputs or stored metadata. The numbers in the constructor names count
boundary legs, not coarse-grained effective legs.
These reference labels do not always use one unit of area per vertex; use
`lattice_geometry` below for a consistent unit-square metric.

Two families extend the named examples:

```julia
row_graph(8; layers=2, shift=0)
diagonal_graph(8; shift=0)
```

Widths must be even and at least two. Row layers must be positive. A positive
shift moves the top of column `j` to output `mod1(j + shift, width)`. Signed
shifts are reduced modulo the width.

Rows use `[left up; down right]` ports. A/B alternate across each row and
between layers. For repeated checkerboard rows, `layers-shift` must be even.
For example, `row_4()` defines a valid contraction for distinct A/B, but its
repeated closure does not preserve the checkerboard pattern unless A=B.

Diagonal graphs apply B on pairs `(1,2), (3,4), ...`, then A on
`(2,3), (4,5), ..., (width,1)`. All named diagonal constructors use this
same convention: `diagonal_2(; A, B)` equals `diagonal_graph(2; A, B)`.

Wide graphs are constructed explicitly. In particular, `shifted_row_8()` does
not perform TNRKit's preliminary coarse graining or introduce a truncation.

## Construct other orientations

Construct each graph directly, passing transformed nodes where needed:

```julia
using TensorKit

A = TensorNode(:A, (2, 2))
B = TensorNode(:B, (2, 2))
p1 = ((3, 1), (4, 2))
p2 = ((2, 4), (1, 3))

vertical = double_row_2(; A, B)
horizontal = double_row_2(; A=permute(B, p1), B=permute(A, p1))
diagonal = TensorLatticeLab.tilted_4(; A, B)
antidiagonal = TensorLatticeLab.tilted_4(; A=permute(B, p2), B=permute(A, p2))
```

These four expressions reproduce TNRKit's 2×2 calibration graphs. Rotations
and color exchange are node views in the graphs. Each graph has four tensor
occurrences and uses the original base A/B bindings. Construction performs no
numerical contraction and makes no planarity assumption.

## Infer unit-square geometry

```julia
geo = lattice_geometry(diagonal_4())
geo.space_period        # (2, 2)
geo.time_period         # (-1, 1)
geo.area                # 4
geo.circumference       # 2sqrt(2)
geo.height              # sqrt(2)
geo.relative_shift      # 0//1 (exact rational)
geo.modular_parameter   # 0.0 + 0.5im
```

Each node represents a 1×1 square with sides `[left up; down right]`. The
function closes output i onto input i and infers the translation periods from
the actual edges, including nested graphs and node views. It uses no tensors.
The result is a `LatticeGeometry` storing only `space_period`, `time_period`
and `positions`. The positions are tile centers in expanded occurrence order
for inspection. Area, lengths, relative shift and tau are computed properties, so
`geo.area` and `geo.modular_parameter` do not duplicate stored data.
`relative_shift` is displacement divided by circumference, calculated exactly
as `dot(space_period,time_period) // dot(space_period,space_period)`.

The time period is defined modulo the space period. The returned basis has
positive area and `-1/2 < real(tau) <= 1/2`, preserving which cycle is spatial.
For example, `shifted_row_4()` gives time period `(-1,1)` and
`tau = (-1+im)/4`: this sign follows the specified port and boundary convention.

For `TensorLatticeLab.tilted_4()`, the inferred periods are `(2,2)` and `(0,2)`:
area 4, circumference `2sqrt(2)`, height `sqrt(2)`, relative shift `1//2`,
and tau `(1+im)/2`.
Those lengths are twice its TNRKit reference shape label. Here each of the four
vertices contributes exactly one unit of area.

This reference metric is distinct from geometry fitted from numerical tensors.
Its `area` is a tile count, not the additive area contribution to log Z.
Gluings that cannot represent a connected flat square torus with the declared
spatial cut are rejected. See [the derivation](docs/lattice-geometry.md) and
[the runnable example](examples/06_lattice_geometry.jl).

## Bind tensors without contracting

```julia
# a and b are existing rank-(2,2) TensorMaps.
A, B = TensorNode(:A, a), TensorNode(:B, b)
G = diagonal_4(; A, B)
bindings = checkerboard_coloring(G, (a, b))
```

`checkerboard_coloring(G, a)` binds both standard colors to the same tensor. The function first checks that every internal bond, including annular closure, joins opposite A/B sites. Upper and lower legs stay open. TensorGraphLab checks ranks, directions and connected
spaces. For other assignments, use explicit bindings such as
`coloring(G, :A => a, :B => b)`.

Bindings retain the original tensors. Validation uses TensorKit space metadata,
including for rotated node views, without transforming tensor data or
materializing the transfer tensor.

`checkerboard_coloring(G, cell)` directly accepts `UniformGates` and
`CheckerboardGates`. The latter maps `black` to A and `white` to B. There is no
default mapping for `PlaquetteGates` or four independent tensors; write the
desired graph and explicit color assignment for those cases.

## Inspect a transfer operator

```julia
tube = diagonal_2()
op = TransferOperator(tube, checkerboard_coloring(tube, (a, b)))
domain(op) == codomain(op)  # Required by construction, not just equal dimensions.
space(op)                 # TensorKit map space, with the graph's boundary order.
numout(op), numin(op), numind(op)
spacetype(op), sectortype(op), scalartype(op)
geo = lattice_geometry(op)
```

The operator stores its tube and coloring. The leading type parameters
`TransferOperator{E,S,M,N}` record the promoted scalar type, elementary space
type, and numbers of physical output and input legs. For example,
`sectortype(typeof(op))` returns `sectortype(S)` directly from the type.
The boundary spaces in `space(op)` come from the colored tube.
`lattice_geometry(op)` forwards to the reference geometry of its graph and
requires the same unit-square gluing assumptions as `lattice_geometry(G)`.
The operator supplies the lattice coloring. Fix the center sector `a` or bulk
leg `V`, supply the operator, then apply the result to the current state.

```julia
ψnext = anyon_insertion_irrep(a)(op)(ψ)   # boundary TensorMap
Unext = anyon_insertion_rep(V)(op)(U)     # CentralTensorMap with bulk leg V
```

To reuse the prepared action, keep `irrep = anyon_insertion_irrep(a, op)`
or `rep = anyon_insertion_rep(V, op)` and call it on successive states.

For a lattice coloring with sector type `I`, the anyon has type
`C <: DrinfeldCenter{I}` and the bulk leg has type `GradedSpace{C}`.
The representation reuses one contraction plan for its occupied sectors and
one irrep across all columns of each sector.

The action implements the left partial trace in
[Eq. (29)](https://arxiv.org/pdf/2610.08787#page=17). Every crossing is
printed as `gamma'`. Coloring resolves the positive half-braiding at the
oriented annular space of each occurrence. Contraction planning treats all
segments of the charge line as O(1) at fixed sector. The lattice tensors,
half-braidings and contraction function are reused across applications.
See [the bounded example](examples/06_anyon_action.jl) for direct graph evaluation.

## Diagonalize a transfer operator

Use the transfer operator, requested center space and solver configuration together.

```julia
C = BoundaryBulkKit.take_center(sectortype(op))
levels = Vect[C](one(C)=>1, a=>3)
cfg = TransferMatrixConfig(which=:LM,
    solver=TransferArnoldi(tol=1e-10, kwargs=(; eager=true)))
spec = eigsolve(op, levels, cfg; rightstates=Dict(one(C) => [1], a => [2, 3]))
λ = block(spec.D, a).diag
ψ = rightstate(spec, a, 2)
```

Here `a` is a nontrivial sector of `C`. Its multiplicity requests three leading
eigenvalues by magnitude. A zero or omitted multiplicity skips the sector,
including the trivial sector. Use `Vect[C](one(C)=>1)` to request only its
leading eigenvalue, or `Vect[C]()` to skip all solves.
Choose right eigenvectors with the optional `rightstates` keyword:

| Selection | Retained right eigenvectors |
| --- | --- |
| `0` (default) | None. Eigenvalues and convergence reports are returned. |
| `:all` | Every requested level in every sector. |
| `N` | Levels 1 through `min(N, dim(levels, a))` in each requested sector. |
| `Dict(a => [2, 3], b => [1])` | Levels 2 and 3 in sector `a` and level 1 in sector `b`. Other sectors retain none. |

Explicit indices must lie within the requested levels in their sector. The
dictionary uses the original one-based eigenvalue order, as does `rightstate`.
In the example, `rightstate(spec, a, 2)` reads column 1 of `block(spec.U, a)`.
When right states are retained, `U` is a `CentralTensorMap` with one bulk leg
in the requested orientation. Its boundary sector is `I = sectortype(op)` and
its center sector type is `C <: DrinfeldCenter{I}`. The default largest-magnitude
solve uses the Schur route when no extra KrylovKit keywords are supplied.
Single-start Arnoldi can miss multiplicities of exactly degenerate eigenvalues.
Infeasible counts and insufficient convergence raise errors.

Use `ExperimentLab.configtree(cfg)` to inspect the configuration and `@config`
to construct it as a tree. `ExperimentLab.writeconfig(path, cfg)` saves the
solver settings and the random seed. Restore them
with `ExperimentLab.readconfig(path, TransferMatrixConfig)` and pass the
desired sector space and right-state selection to each call to `eigsolve`.

## Memory and examples

Do **not** materialize a complete transfer tensor to try these graphs on
production data. With `w` legs on each boundary and uniform bond dimension D,
an unconstrained dense transfer requires D^(2w) entries. Even an action on a
state can have large intermediates. Spectrum tests use small sector state
spaces, known eigenvalues and independently checked right-state residuals.

Open any of these files in VS Code and run it in full; each activates the
package relative to its own directory:

1. [`01_diagonal_graph.jl`](examples/01_diagonal_graph.jl): explicit `@tube`
   compared with `diagonal_4()`.
2. [`02_shifted_row.jl`](examples/02_shifted_row.jl): shifted four- and eight-leg
   rows, with no numerical contraction.
3. [`03_oriented_graphs.jl`](examples/03_oriented_graphs.jl): construct and
   inspect four graph orientations.
4. [`04_coloring.jl`](examples/04_coloring.jl): validate A/B bindings using
   one-entry tensors.
5. [`05_state_graph.jl`](examples/05_state_graph.jl) composes a symbolic action on X.
6. [`06_anyon_action.jl`](examples/06_anyon_action.jl) evaluates a small center-sector action.
6. [`06_lattice_geometry.jl`](examples/06_lattice_geometry.jl): infer periods,
   area, circumference, height, relative shift and tau from five graph constructions.
7. [`07_tensor_cells.jl`](examples/07_tensor_cells.jl): construct periodic cells
   and bind their tensors to a graph, independently of LoopTNRLab.
8. [`08_transfer_operator.jl`](examples/08_transfer_operator.jl): inspect boundary
   spaces, tensor traits and reference geometry without applying the operator.

9. [`09_transfer_spectrum.jl`](examples/09_transfer_spectrum.jl) runs a small ED configuration and checks a right eigenstate.

The development environment uses the sibling checkouts specified in `Project.toml`. Run tests with
`julia --project=. -e 'using Pkg; Pkg.test()'` to include test dependencies.

## Reference conventions

The small transfer and auxiliary expressions are based on TNRKit's
`src/utility/transfer_matrix.jl` and `_extract_tau_and_c_2x2` in
`src/utility/cft.jl`. The graph families expose those connections directly;
they do not depend on TNRKit or reuse its numerical shape-based dispatch.
The two-leg diagonal uses the same B-then-A convention as the rest of this
package's diagonal family, rather than TNRKit's A-then-B convention.
