@testset "UniformGates — periodic 1×1 indexing" begin
    gate = test_gate()
    cell = UniformGates(gate)

    @test cell isa AbstractTensorCell
    @test cell isa AbstractMatrix
    @test size(cell) == (1, 1)
    @test axes(cell) == (Base.OneTo(1), Base.OneTo(1))
    @test length(cell) == 1
    @test cell[1, 1] === gate
    @test cell[-3, 4] === gate
    @test only(cell) === gate
    @test assertconsistency(cell) === nothing
end

@testset "CheckerboardGates — staggered periodic indexing" begin
    black = test_gate(west=ℂ^2, south=ℂ^3, north=ℂ^4, east=ℂ^5)
    white = test_gate(west=ℂ^5, south=ℂ^4, north=ℂ^3, east=ℂ^2)
    cell = CheckerboardGates(black, white)

    @test size(cell) == (1, 2)
    @test cell[1, 1] === black
    @test cell[1, 2] === white
    @test cell[2, 1] === white
    @test cell[2, 2] === black
    @test cell[3, 3] === black
    @test cell[0, 0] === black
    @test cell[-1, -1] === black
    @test collect(cell) == [black white]
    @test assertconsistency(cell) === nothing
end

@testset "PlaquetteGates — periodic 2×2 indexing" begin
    northwest = test_gate()
    northeast = test_gate()
    southwest = test_gate()
    southeast = test_gate()
    matrix = [northwest northeast; southwest southeast]

    from_matrix = PlaquetteGates(matrix)
    shared_matrix = PlaquetteGates(matrix; copy=false)
    from_entries = PlaquetteGates(
        northwest, northeast, southwest, southeast
    )

    @test size(from_matrix) == (2, 2)
    @test from_matrix.northwest !== northwest
    @test from_matrix.northeast !== northeast
    @test from_matrix.southwest !== southwest
    @test from_matrix.southeast !== southeast
    @test shared_matrix.northwest === northwest
    @test shared_matrix.northeast === northeast
    @test shared_matrix.southwest === southwest
    @test shared_matrix.southeast === southeast
    @test collect(from_matrix) == matrix
    @test collect(from_entries) == matrix
    @test shared_matrix[1, 1] === northwest
    @test shared_matrix[1, 2] === northeast
    @test shared_matrix[2, 1] === southwest
    @test shared_matrix[2, 2] === southeast
    @test shared_matrix[3, 4] === northeast
    @test shared_matrix[0, -1] === southwest
    @test assertconsistency(from_matrix) === nothing

    @test_throws DimensionMismatch PlaquetteGates(fill(northwest, 1, 2))
end

@testset "gate cells — copy preserves topology without aliases" begin
    gate = test_gate()
    uniform = UniformGates(gate)
    checkerboard = CheckerboardGates(gate, gate)
    plaquette = PlaquetteGates(fill(gate, 2, 2); copy=false)

    for cell in (uniform, checkerboard, plaquette)
        copied = copy(cell)

        @test typeof(copied) == typeof(cell)
        @test collect(copied) == collect(cell)
        @test all(copied[index] !== cell[index] for index in CartesianIndices(cell))
        @test all(
            copied[index].data !== cell[index].data
            for index in CartesianIndices(cell)
        )
        @test length(unique(objectid.(collect(copied)))) == length(copied)
    end
end

@testset "gate cells promote to larger unit cells" begin
    gate = test_gate()
    uniform = UniformGates(gate)
    checkerboard = CheckerboardGates(uniform)
    uniform_plaquette = PlaquetteGates(uniform)

    @test checkerboard.black !== gate
    @test checkerboard.white !== gate
    @test checkerboard.black !== checkerboard.white
    @test checkerboard.black ≈ gate
    @test checkerboard.white ≈ gate
    @test all(entry !== gate for entry in uniform_plaquette)
    @test all(entry ≈ gate for entry in uniform_plaquette)
    @test length(unique(objectid.(collect(uniform_plaquette)))) == 4

    shared_checkerboard = CheckerboardGates(uniform; copy=false)
    shared_uniform_plaquette = PlaquetteGates(uniform; copy=false)

    @test shared_checkerboard.black === gate
    @test shared_checkerboard.white === gate
    @test all(entry === gate for entry in shared_uniform_plaquette)

    black = test_gate()
    white = test_gate()
    checkerboard = CheckerboardGates(black, white)
    plaquette = PlaquetteGates(checkerboard)

    @test plaquette.northwest !== black
    @test plaquette.northeast !== white
    @test plaquette.southwest !== white
    @test plaquette.southeast !== black
    @test plaquette.northwest !== plaquette.southeast
    @test plaquette.northeast !== plaquette.southwest
    @test collect(plaquette) == [black white; white black]

    shared_plaquette = PlaquetteGates(checkerboard; copy=false)

    @test shared_plaquette.northwest === black
    @test shared_plaquette.northeast === white
    @test shared_plaquette.southwest === white
    @test shared_plaquette.southeast === black

    @test PlaquetteGates(plaquette; copy=false) === plaquette
    @test PlaquetteGates(plaquette) !== plaquette
end

@testset "gate cells require square four-leg tensors" begin
    mps_tensor = randn(ComplexF64, ℂ^2 ⊗ ℂ^3, ℂ^4)
    @test_throws MethodError UniformGates(mps_tensor)
    @test_throws MethodError CheckerboardGates(mps_tensor, mps_tensor)
    @test_throws MethodError PlaquetteGates(fill(mps_tensor, 2, 2))
end
