include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab, TensorLatticeLab

let
    # Single-entry fixtures keep this example independent of numerical solvers.
    gate = ones(ComplexF64, ℂ^1 ⊗ ℂ^1 ← ℂ^1 ⊗ ℂ^1)
    uniform = UniformGates(gate)
    checkerboard = CheckerboardGates(gate, 2gate)
    plaquette = PlaquetteGates(checkerboard)
    for cell in (uniform, checkerboard, plaquette)
        @assert isnothing(TensorLatticeLab.assertconsistency(cell))
        show(stdout, MIME"text/plain"(), cell)
        println("\n")
    end
    bindings = checkerboard_coloring(diagonal_4(), checkerboard)
    @assert bindings[:A] === checkerboard.black
    @assert bindings[:B] === checkerboard.white
    println("Example 7 passed: cell construction and coloring without LoopTNRLab.")
end
