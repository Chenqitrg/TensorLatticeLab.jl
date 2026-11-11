# Exploratory regression: the composition checks deliberately expose a failure
# of shortest-cut routing. This is not the public prepare_action interface.
include(joinpath(@__DIR__, "..", "..", "examples", "_setup.jl"))
using Test, TensorKit, TensorGraphLab, TensorLatticeLab
import BoundaryBulkKit

Base.include(TensorLatticeLab, joinpath(@__DIR__, "routing.jl"))
Base.include(TensorLatticeLab, joinpath(@__DIR__, "action.jl"))
const prepare_action = TensorLatticeLab.prepare_action
include(joinpath(@__DIR__, "checks.jl"))
