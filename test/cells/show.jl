@testset "gate cells — plain-text display" begin
    gate = test_gate()
    uniform = UniformGates(gate)
    uniform_summary = "1×1 UniformGates(ComplexF64, ComplexSpace)"

    @test summary(uniform) == uniform_summary
    @test sprint(show, uniform) == uniform_summary
    @test sprint(
        show, MIME"text/plain"(), uniform; context=(:compact => true,)
    ) == uniform_summary

    uniform_shown = sprint(show, MIME"text/plain"(), uniform)
    @test startswith(uniform_shown, uniform_summary * ":\n")
    @test occursin("    ⋮\n    │\n…─[1,1]─…\n    │\n    ⋮", uniform_shown)
    @test occursin("[1,1] W=ℂ^2, S=ℂ^3, N=ℂ^3, E=ℂ^2", uniform_shown)
    @test !occursin("TensorMap", uniform_shown)

    checkerboard = CheckerboardGates(gate, gate)
    checkerboard_shown = sprint(show, MIME"text/plain"(), checkerboard)
    @test startswith(
        checkerboard_shown,
        "1×2 CheckerboardGates(ComplexF64, ComplexSpace):\n",
    )
    @test occursin("…─[1,1]─────[1,2]─…", checkerboard_shown)
    @test occursin("…─[2,1]─────[2,2]─…", checkerboard_shown)

    plaquette = PlaquetteGates(fill(gate, 2, 2))
    plaquette_shown = sprint(show, MIME"text/plain"(), plaquette)
    @test startswith(
        plaquette_shown,
        "2×2 PlaquetteGates(ComplexF64, ComplexSpace):\n",
    )
    @test occursin("…─[1,1]─────[1,2]─…", plaquette_shown)
    @test occursin("…─[2,1]─────[2,2]─…", plaquette_shown)
end
