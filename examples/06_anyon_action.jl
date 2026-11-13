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

    irrep = anyon_insertion_irrep(anyon)(op)
    showgraph(irrep.graph)
    result = irrep(ψ)

    C = typeof(anyon)
    U = BoundaryBulkKit.CentralTensorMap(copy(ψ.data), domain(op), Vect[C](anyon => 1))
    rep = anyon_insertion_rep(only(domain(U)))(op)
    @assert rep(U).data ≈ result.data
    println("Output state norm: ", norm(result))
end
