@testset "LatticeGeometry result" begin
    geo = LatticeGeometry((2, 0), (1, 1), [(0, 0), (1, 0)])
    @test geo.area == 2
    @test geo.modular_parameter ≈ 0.5 + 0.5im
    @test geo.circumference == 2
    @test geo.height == 1
    @test geo.relative_shift === 1//2
    @test hasproperty(geo, :space_period) && hasproperty(geo, :modular_parameter)
    @test occursin("Space period:", sprint(show, MIME"text/plain"(), geo))
    @test_throws ArgumentError LatticeGeometry((1, 0), (2, 0), [(0, 0)])
    @test_throws ArgumentError LatticeGeometry((1, 0), (0, -1), [(0, 0)])
    @test_throws ArgumentError LatticeGeometry((2, 0), (0, 2), [(0, 0)])
end
