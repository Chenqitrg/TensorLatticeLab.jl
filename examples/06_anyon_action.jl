include("_setup.jl")
using TensorKit, TensorGraphLab, TensorLatticeLab
import BoundaryBulkKit

let
    V = Z3Space(0 => 1, 1 => 1, 2 => 1)
    A = randn(ComplexF64, V ⊗ V ← V ⊗ V)
    tube = diagonal_2()
    op = TransferOperator(tube, checkerboard_coloring(tube, (A, A)))
    anyon = last(collect(values(BoundaryBulkKit.take_center(Z3Irrep))))
    X = BoundaryBulkKit.forget(anyon)
    ψ = randn(ComplexF64, domain(op) ← X)

    G = action_graph(tube)
    showgraph(G)
    gamma(W::ElementarySpace) = BoundaryBulkKit.HalfBraiding(anyon, W)
    prepared = prepare_action(G, op.coloring; gamma)
    result = prepared(ψ)
    @assert result ≈ evaluate(G, coloring(prepared, ψ); method=:planar, opt=true)

    apply = prepare_action(op, anyon)
    @assert apply(ψ) ≈ result
    println("Output state norm: ", norm(result))
end
