# Run the public action against the independent, bounded transfer references.
include(joinpath(@__DIR__, "..", "..", "examples", "_setup.jl"))
using Test, TensorKit, TensorGraphLab, TensorLatticeLab
include(joinpath(@__DIR__, "..", "..", "test", "transfers", "action.jl"))
