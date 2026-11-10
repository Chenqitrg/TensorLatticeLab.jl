include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab, TensorLatticeLab

let
    transfer = diagonal_4()
    X = TensorNode(:X, (4, 1))
    @graph action[o1 o2 o3 o4; c] :=
        transfer[o1 o2 o3 o4; p1 p2 p3 p4] * X[p1 p2 p3 p4; c]
    @assert numout(action) == 4 && numin(action) == 1
    showgraph(action; name=:action)
    println("Example 5 passed: a symbolic action on X, without allocating a state.")
    # Center-sector insertion is not implemented yet. This symbolic example
    # does not claim to be the planar anyon action required by the eigensolver.
end
