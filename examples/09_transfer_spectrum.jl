include("_setup.jl")
using TensorKit, TensorGraphLab, TensorLatticeLab
using ExperimentLab: @config
import BoundaryBulkKit
import ExperimentLab
import Random

let
    V = Z3Space(0=>1, 1=>1, 2=>1)
    A = randn(Random.Xoshiro(13), ComplexF64, V ⊗ V ← V ⊗ V)
    tube = diagonal_2()
    op = TransferOperator(tube, checkerboard_coloring(tube, (A,A)))
    C = BoundaryBulkKit.take_center(sectortype(op))
    charge = C(1, 0)
    levels = Vect[C](one(C)=>1, charge=>2)

    cfg = @config """
    TransferMatrixConfig
    ├─ trunc: truncspace(levels; by=abs, rev=true)
    ├─ rightstates: Dict(one(C) => [1], charge => [2])
    └─ solver: TransferArnoldi
       └─ tol: 1.0e-10
    """
    println(ExperimentLab.configtree(cfg))
    spec = eigsolve(op, cfg)
    λ = block(spec.D, charge).diag
    ψ = rightstate(spec, charge, 2)
    @assert length(λ) == 2
    @assert anyon_insertion_irrep(charge, op)(ψ) ≈ λ[2] * ψ
    println("Leading transfer eigenvalues: ", λ)
end
