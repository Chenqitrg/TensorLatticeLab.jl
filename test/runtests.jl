using Test, TensorKit, TensorGraphLab, TensorLatticeLab
using TensorGraphLab: occurrences, expand, boundary, dualflags, closed_graph

include("cells/fixtures.jl")
include("cells/types.jl")
include("cells/bonds.jl")
include("cells/constructors.jl")
include("cells/cdl.jl")
include("cells/show.jl")

include("transfers/graphs.jl")
include("transfers/coloring.jl")
include("transfers/operator.jl")
include("transfers/action.jl")
include("ED/spectrum.jl")

include("geometry/types.jl")
include("geometry/inference.jl")
include("package_boundary.jl")
