include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab, TensorLatticeLab

let
    # Each fixture contains exactly one numerical entry. This example only
    # checks bindings; it never materializes a transfer tensor.
    V = ℂ^1
    a = TensorMap(ones(ComplexF64, 1, 1), V ⊗ V ← V ⊗ V)
    b = 2a
    graph = diagonal_4()
    bindings = coloring(graph, (a, b))
    @assert bindings[:A] === a
    @assert bindings[:B] === b
    @assert isnothing(TensorGraphLab.validate(graph, bindings))
    uniform = coloring(graph, a)
    @assert uniform[:A] === uniform[:B] === a
    println("Example 4 passed: default A/B bindings, without contraction.")
end
