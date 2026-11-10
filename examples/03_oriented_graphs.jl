include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab, TensorLatticeLab

let
    A = TensorNode(:A, (2, 2))
    B = TensorNode(:B, (2, 2))
    p1 = ((3, 1), (4, 2))
    p2 = ((2, 4), (1, 3))

    vertical = double_row_2(; A, B)
    horizontal = double_row_2(; A=permute(B, p1), B=permute(A, p1))
    diagonal = TensorLatticeLab.tilted_4(; A, B)
    antidiagonal = TensorLatticeLab.tilted_4(; A=permute(B, p2), B=permute(A, p2))

    showgraph(vertical; name=:vertical)
    showgraph(horizontal; name=:horizontal)
    showgraph(diagonal; name=:diagonal)
    showgraph(antidiagonal; name=:antidiagonal)
    println("Example 3 passed: construct each orientation directly from node views.")
end
