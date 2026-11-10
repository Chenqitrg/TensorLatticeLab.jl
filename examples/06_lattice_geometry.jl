include(joinpath(@__DIR__, "_setup.jl"))
using TensorLatticeLab

let
    for (name, graph) in (
        :diagonal_2 => diagonal_2(),
        :diagonal_4 => diagonal_4(),
        :double_row_2 => double_row_2(),
        :shifted_row_4 => shifted_row_4(),
        :tilted_4 => TensorLatticeLab.tilted_4(),
    )
        geo = lattice_geometry(graph)
        println(name)
        show(stdout, MIME"text/plain"(), geo)
        println("\n")
    end
    geo = lattice_geometry(TensorLatticeLab.tilted_4())
    @show geo
    @assert geo.area == 4
    @assert geo.relative_shift === 1//2
    @assert geo.modular_parameter ≈ 0.5 + 0.5im
    println("Example 6 passed: geometry inferred from unit-square gluing, without tensors.")
end
