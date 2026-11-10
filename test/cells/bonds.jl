@testset "gate leg multiplication" begin
    west, south, north, east = ℂ^2, ℂ^3, ℂ^4, ℂ^5
    new_west, new_south, new_north, new_east = ℂ^6, ℂ^7, ℂ^8, ℂ^9
    gate = randn(ComplexF64, west ⊗ south, north ⊗ east)
    west_matrix = randn(ComplexF64, new_west ← west)
    south_matrix = randn(ComplexF64, new_south ← south)
    north_matrix = randn(ComplexF64, north ← new_north)
    east_matrix = randn(ComplexF64, east ← new_east)

    west_result = mul_west(west_matrix, gate)
    south_result = mul_south(south_matrix, gate)
    north_result = mul_north(gate, north_matrix)
    east_result = mul_east(gate, east_matrix)

    @test space(west_result) == (new_west ⊗ south ← north ⊗ east)
    @test space(south_result) == (west ⊗ new_south ← north ⊗ east)
    @test space(north_result) == (west ⊗ south ← new_north ⊗ east)
    @test space(east_result) == (west ⊗ south ← north ⊗ new_east)
    @test west_result ≈ (west_matrix ⊗ id(ComplexF64, south)) * gate
    @test south_result ≈ (id(ComplexF64, west) ⊗ south_matrix) * gate
    @test north_result ≈ gate * (north_matrix ⊗ id(ComplexF64, east))
    @test east_result ≈ gate * (id(ComplexF64, north) ⊗ east_matrix)
end

@testset "gate cells standardize every periodic bond" begin
    semion_category = CategoryData.PMFC{2, 1, 0, 1, 1, 1}
    sector_type = CategoryData.Object{semion_category}
    vacuum = sector_type(1)
    semion = sector_type(2)
    @test frobenius_schur_indicator(semion) == -1
    nondual = Vect[sector_type](vacuum => 2, semion => 2)
    dual_bond = dual(nondual)
    rng = Xoshiro(0xce11)

    raw_uniform = rand(
        rng, ComplexF64, dual_bond ⊗ dual_bond ← dual_bond ⊗ dual_bond
    )
    raw_uniform_space = space(raw_uniform)
    uniform = UniformGates(raw_uniform)

    @test space(raw_uniform) == raw_uniform_space
    @test all_nondual(uniform)
    @test norm(uniform.gate) ≈ norm(raw_uniform)
    @test assertconsistency(uniform) === nothing

    # Only the black-east/white-west bond is dual. Its two explicit gauge
    # matrices must cancel even for Semion, whose FS indicator is nontrivial.
    trivial = Vect[sector_type](vacuum => 1)
    raw_black = rand(
        rng, ComplexF64, nondual ⊗ trivial ← trivial ⊗ dual_bond
    )
    raw_white = rand(
        rng, ComplexF64, dual_bond ⊗ trivial ← trivial ⊗ nondual
    )
    raw_spaces = (space(raw_black), space(raw_white))
    checkerboard = CheckerboardGates(raw_black, raw_white)

    @plansor before[-1 -2 -4; -3 -5 -6] :=
        raw_black[-1 -2; -3 1] * raw_white[1 -4; -5 -6]
    @plansor after[-1 -2 -4; -3 -5 -6] :=
        checkerboard.black[-1 -2; -3 1] *
        checkerboard.white[1 -4; -5 -6]

    @test space(raw_black) == raw_spaces[1]
    @test space(raw_white) == raw_spaces[2]
    @test all_nondual(checkerboard)
    @test after ≈ before

    raw_upper = rand(
        rng, ComplexF64, trivial ⊗ dual_bond ← nondual ⊗ trivial
    )
    raw_lower = rand(
        rng, ComplexF64, trivial ⊗ nondual ← dual_bond ⊗ trivial
    )
    vertical = CheckerboardGates(raw_upper, raw_lower)
    @plansor vertical_before[-1 -4 -5; -2 -3 -6] :=
        raw_upper[-1 1; -2 -3] * raw_lower[-4 -5; 1 -6]
    @plansor vertical_after[-1 -4 -5; -2 -3 -6] :=
        vertical.black[-1 1; -2 -3] *
        vertical.white[-4 -5; 1 -6]

    @test all_nondual(vertical)
    @test vertical_after ≈ vertical_before

    raw_plaquette = [
        rand(
            rng,
            ComplexF64,
            dual_bond ⊗ dual_bond ← dual_bond ⊗ dual_bond,
        ) for _ in 1:2, _ in 1:2
    ]
    raw_plaquette_spaces = space.(raw_plaquette)
    plaquette = PlaquetteGates(raw_plaquette; copy=false)

    @test space.(raw_plaquette) == raw_plaquette_spaces
    @test all_nondual(plaquette)
    @test norm.(collect(plaquette)) ≈ norm.(raw_plaquette)
    @test assertconsistency(plaquette) === nothing

    for cell in (uniform, checkerboard, plaquette)
        standardized = standardize(cell)
        @test standardized !== cell
        @test typeof(standardized) == typeof(cell)
        @test all_nondual(standardized)
        @test all(
            standardized[index] ≈ cell[index]
            for index in CartesianIndices(cell)
        )
    end
end

@testset "gate cells reject open periodic bonds" begin
    open_uniform = test_gate(east=ℂ^4)
    @test_throws TensorKit.SpaceMismatch UniformGates(open_uniform)

    black = test_gate(west=ℂ^2, south=ℂ^3, north=ℂ^4, east=ℂ^5)
    wrong_black_neighbor = test_gate(
        west=ℂ^6, south=ℂ^4, north=ℂ^3, east=ℂ^2
    )
    wrong_white_neighbor = test_gate(
        west=ℂ^5, south=ℂ^4, north=ℂ^3, east=ℂ^6
    )
    @test_throws TensorKit.SpaceMismatch CheckerboardGates(
        black, wrong_black_neighbor
    )
    @test_throws TensorKit.SpaceMismatch CheckerboardGates(
        black, wrong_white_neighbor
    )

    compatible = test_gate()
    incompatible_northeast = test_gate(west=ℂ^4, east=ℂ^4)
    plaquette = [compatible incompatible_northeast; compatible compatible]
    @test_throws TensorKit.SpaceMismatch PlaquetteGates(plaquette)
end
