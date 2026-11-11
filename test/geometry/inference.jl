@testset "Unit-square lattice geometry" begin
    for (graph, area, circumference, height, relative_shift, tau) in (
        (diagonal_2(), 2, sqrt(2), sqrt(2), 0//1, 1im),
        (diagonal_4(), 4, 2sqrt(2), sqrt(2), 0//1, 0.5im),
        (double_row_2(), 4, 2, 2, 0//1, 1im),
        (shifted_row_2(), 2, 2, 1, 1//2, 0.5 + 0.5im),
        (row_4(), 4, 4, 1, 0//1, 0.25im),
        (shifted_row_4(), 4, 4, 1, -1//4, -0.25 + 0.25im),
        (shifted_row_8(), 8, 8, 1, -1//8, -0.125 + 0.125im),
        (TensorLatticeLab.tilted_4(), 4, 2sqrt(2), sqrt(2), 1//2, 0.5 + 0.5im),
    )
        geo = lattice_geometry(graph)
        @test geo.area == area
        @test geo.circumference ≈ circumference
        @test geo.height ≈ height
        @test geo.relative_shift === relative_shift
        @test geo.modular_parameter ≈ tau
        @test length(unique(geo.positions)) == area
        @test geo.circumference * geo.height ≈ area
        s, t = geo.space_period, geo.time_period
        @test s[1]*t[2] - s[2]*t[1] == area
    end

    @test lattice_geometry(row_graph(16; layers=3, shift=5)).area == 48
    # An exact fraction that has no finite binary floating-point representation.
    @test lattice_geometry(row_graph(6; shift=1)).relative_shift === -1//6
    A, B = TensorNode(:A, (2, 2)), TensorNode(:B, (2, 2))
    g = closed_graph(diagonal_4())
    @test lattice_geometry(substitute(g, A=>B, B=>A)).space_period == lattice_geometry(g).space_period
    @test lattice_geometry(substitute(g, A=>B, B=>A)).time_period == lattice_geometry(g).time_period

    # Global rotations change the vectors, not their relative geometry.
    p = ((3, 1), (4, 2))
    rotated = substitute(g, A=>permute(A, p), B=>permute(B, p))
    @test lattice_geometry(rotated).modular_parameter ≈ lattice_geometry(g).modular_parameter
    @test lattice_geometry(rotated).area == 4
    # Adjoint reverses the cut direction without moving the base square sides.
    # Positive orientation reverses both periods, leaving their ratio unchanged.
    original = lattice_geometry(shifted_row_4())
    reversed = lattice_geometry(closed_graph(shifted_row_4())')
    @test reversed.space_period == .-original.space_period
    @test reversed.time_period == .-original.time_period
    @test reversed.modular_parameter ≈ original.modular_parameter

    # Grouping and naming have no geometric significance.
    @graph nested[a b c d; e f h j] := g[a b c d; e f h j]
    @test lattice_geometry(nested).time_period == lattice_geometry(g).time_period
    @test lattice_geometry(nested).space_period == lattice_geometry(g).space_period
    @graph reordered[o1 o2 o3 o4; p1 p2 p3 p4] :=
        B[i3 i4; p3 p4] * A[o4 o1; i4 i1] *
        B[i1 i2; p1 p2] * A[o2 o3; i2 i3]
    @test lattice_geometry(reordered).space_period == lattice_geometry(g).space_period
    @test lattice_geometry(reordered).time_period == lattice_geometry(g).time_period

    # A single tile illustrates the scale of the four-tile tilted graph.
    @graph tile[o1 o2; p1 p2] := A[o2 o1; p1 p2]
    @test lattice_geometry(tile).area == 1
    @test lattice_geometry(tile).modular_parameter ≈ 0.5 + 0.5im
end

@testset "Reject incompatible square-torus gluings" begin
    @test_throws ArgumentError lattice_geometry(diagonal_graph(4; shift=1))
    A = TensorNode(:A, (2, 2))
    # Horizontal and vertical translations do not commute.
    @graph nonflat[o1 o2 o3; p1 p2 p3] :=
        A[h3 o1; p2 h1] * A[h1 o2; p1 h2] * A[h2 o3; p3 h3]
    @test_throws ArgumentError lattice_geometry(nonflat)
    # Opening just one edge in a 2x2 torus is not a closed spatial cut.
    @graph broken_cut[o; p] :=
        A[h2 o; v31 h1] * A[h1 v24; v42 h2] *
        A[h4 v31; p h3] * A[h3 v42; v24 h4]
    @test_throws ArgumentError lattice_geometry(broken_cut)
    g = closed_graph(shifted_row_2())
    @graph disconnected[o1 o2 o3 o4; p1 p2 p3 p4] :=
        g[o1 o2; p1 p2] * g[o3 o4; p3 p4]
    @test_throws ArgumentError lattice_geometry(disconnected)
    # Two disconnected strips joined in sequence by the boundary permutation
    # cross the declared seam twice per primitive temporal cycle.
    @graph double_cut[o1 o2 o3 o4; p1 p2 p3 p4] :=
        g[o3 o4; p1 p2] * g[o1 o2; p3 p4]
    @test_throws ArgumentError lattice_geometry(double_cut)
    M = TensorNode(:M, (1, 1))
    @graph matrix[o; p] := M[o; p]
    @test_throws ArgumentError lattice_geometry(matrix)
    @graph scalar = A[a b; b a]
    @test_throws ArgumentError lattice_geometry(scalar)
end
