@testset "gate cells — random generation" begin
    uniform_space = ℂ^2 ⊗ ℂ^3 ← ℂ^3 ⊗ ℂ^2
    uniform = rand(UniformGates, ComplexF64, uniform_space)
    seeded_uniform = rand(Xoshiro(1), UniformGates, Float64, uniform_space)
    replayed_uniform = rand(Xoshiro(1), UniformGates, Float64, uniform_space)

    @test eltype(uniform.gate) == ComplexF64
    @test space(uniform.gate) == uniform_space
    @test seeded_uniform.gate == replayed_uniform.gate

    blackspace = ℂ^2 ⊗ ℂ^3 ← ℂ^4 ⊗ ℂ^5
    checkerboard = rand(
        Xoshiro(2), CheckerboardGates, Float64, blackspace
    )

    @test eltype(checkerboard.black) == Float64
    @test space(checkerboard.black) == blackspace
    @test codomain(checkerboard.white) == reverse(domain(blackspace))
    @test domain(checkerboard.white) == reverse(codomain(blackspace))
    @test checkerboard.black !== checkerboard.white

    horizontal_ab = ℂ^2
    horizontal_ba = ℂ^3
    horizontal_cd = ℂ^4
    horizontal_dc = ℂ^5
    vertical_ac = ℂ^2
    vertical_ca = ℂ^3
    vertical_bd = ℂ^4
    vertical_db = ℂ^5
    northwest_space =
        horizontal_ba ⊗ vertical_ac ← vertical_ca ⊗ horizontal_ab
    northeast_space =
        horizontal_ab ⊗ vertical_bd ← vertical_db ⊗ horizontal_ba
    southwest_space =
        horizontal_dc ⊗ vertical_ca ← vertical_ac ⊗ horizontal_cd
    southeast_space =
        horizontal_cd ⊗ vertical_db ← vertical_bd ⊗ horizontal_dc
    plaquette_spaces = [
        northwest_space northeast_space; southwest_space southeast_space
    ]
    plaquette = rand(PlaquetteGates, ComplexF64, plaquette_spaces)

    @test eltype(first(plaquette)) == ComplexF64
    @test space.(collect(plaquette)) == plaquette_spaces
    @test_throws DimensionMismatch rand(
        PlaquetteGates, ComplexF64, fill(uniform_space, 1, 2)
    )
end
