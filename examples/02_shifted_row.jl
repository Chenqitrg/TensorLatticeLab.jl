include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab, TensorLatticeLab

let
    A = TensorNode(:A, (2, 2))
    B = TensorNode(:B, (2, 2))
    @tube annular=(a, r) transfer[o1 o2 o3 o4; p1 p2 p3 p4] :=
        A[a o2; p1 b] * B[b o3; p2 c] * A[c o4; p3 d] * B[d r; p4 a] * id[o1; r]
    @assert transfer == shifted_row_4()
    showgraph(transfer; name=:shifted_row)
    wide = shifted_row_8()
    @assert numout(wide) == numin(wide) == 8
    showgraph(wide; name=:wide_row)
    println("Example 2 passed: shifted rows, without numerical contraction.")
end
