include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab, TensorLatticeLab

let
    # Single-entry tensors suffice for inspecting the operator's static properties.
    a = ones(ComplexF64, ℂ^1 ⊗ ℂ^1 ← ℂ^1 ⊗ ℂ^1)
    tube = diagonal_2()
    bindings = checkerboard_coloring(tube, (a, 2a))
    op = TransferOperator(tube, bindings)

    @assert domain(op) == codomain(op)
    @assert numin(op) == numout(op) == 2
    @assert sectortype(op) === Trivial
    @assert scalartype(op) === ComplexF64
    @assert !applicable(op, a)

    println(op)
    println("Domain: ", domain(op))
    println("Codomain: ", codomain(op))
    println("Map space: ", space(op))
    show(stdout, MIME"text/plain"(), lattice_geometry(op))
    println("\nExample 8 passed: static queries without applying the operator.")
end
