include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab, TensorLatticeLab

let
    A = TensorNode(:A, (2, 2))
    B = TensorNode(:B, (2, 2))
    @tube annular=(i1, r) transfer[o1 o2 o3 o4; p1 p2 p3 p4] :=
        A[o2 o3; i2 i3] * A[o4 r; i4 i1] *
        B[i1 i2; p1 p2] * B[i3 i4; p3 p4] * id[o1; r]
    @assert transfer == diagonal_4()
    showgraph(transfer; name=:transfer)
    println("Example 1 passed: the named graph matches the explicit expression.")
end
