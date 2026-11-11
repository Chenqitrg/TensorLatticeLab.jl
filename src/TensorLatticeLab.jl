module TensorLatticeLab

using TensorKit
import Random
import BoundaryBulkKit
using TensorGraphLab

using TensorGraphLab: closed_graph, dualflags
using TensorGraphLab: AbstractTensorNode, Occurrence, occurrences, expand, base_node, label
using TensorGraphLab: AdjointNode, PermutedNode, TransposedNode, boundary

# Periodic tensor cells and their local tensor operations.
export AbstractTensorCell, UniformGates, CheckerboardGates, PlaquetteGates
export mul_west, mul_south, mul_north, mul_east
export cdl_tensor, random_cdl_tensor
public assertconsistency, standardize

include("cells/types.jl")
include("cells/bonds.jl")
include("cells/constructors.jl")
include("cells/cdl.jl")
include("cells/show.jl")

# Graph families and commonly used small transfers.
export TransferOperator, checkerboard_coloring
export row_graph, diagonal_graph
export diagonal_2, shifted_row_2, double_row_2, diagonal_4
export row_4, shifted_row_4, shifted_row_8

include("transfers/patches.jl")
include("transfers/graphs.jl")
include("transfers/coloring.jl")
include("transfers/operator.jl")

# Reference lattice geometry inferred from graph connections.
export LatticeGeometry, lattice_geometry

include("geometry/types.jl")
include("geometry/inference.jl")

end
