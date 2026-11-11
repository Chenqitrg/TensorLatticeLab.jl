@testset "Transfer graph library" begin
    A = TensorNode(:A, (2, 2))
    B = TensorNode(:B, (2, 2))
    @graph expected[o1 o2; p1 p2] :=
        A[a o2; p1 b] * B[b o1; p2 a]
    @test closed_graph(shifted_row_2()) == expected

    @graph expected[o1 o2; p1 p2] :=
        B[h o1; q1 k] * A[k o2; q2 h] *
        A[l q1; p1 m] * B[m q2; p2 l]
    @test closed_graph(double_row_2()) == expected

    @graph expected[o1 o2; p1 p2] := A[o2 o1; i2 i1] * B[i1 i2; p1 p2]
    @test closed_graph(diagonal_2()) == expected

    @graph expected[o1 o2 o3 o4; p1 p2 p3 p4] :=
        A[o2 o3; i2 i3] * A[o4 o1; i4 i1] *
        B[i1 i2; p1 p2] * B[i3 i4; p3 p4]
    @test closed_graph(diagonal_4()) == expected

    @graph expected[o1 o2 o3 o4; p1 p2 p3 p4] :=
        A[a o2; p1 b] * B[b o3; p2 c] * A[c o4; p3 d] * B[d o1; p4 a]
    @test closed_graph(shifted_row_4()) == expected

    @graph expected[o1 o2 o3 o4; p1 p2 p3 p4] :=
        A[a o1; p1 b] * B[b o2; p2 c] * A[c o3; p3 d] * B[d o4; p4 a]
    @test closed_graph(row_4()) == expected

    @test row_graph(4; shift=-1) == row_graph(4; shift=3)
    @test row_graph(4; shift=5) == shifted_row_4()
    @test diagonal_graph(4; shift=-1) == diagonal_graph(4; shift=3)
    @test diagonal_graph(2) == diagonal_2()
    # The two opposite rotations of the original planar action cross the
    # first middle wire and the first output. The latter has a bare segment.
    @tube annular=(i1, r) diag[o1 o2; p1 p2] := A[o2 r; i2 i1] * B[i1 i2; p1 p2] * id[o1; r]
    @test diag == diagonal_2()
    @test length(diag.annular) == 2
    @test occursin("id[", sprint(showgraph, diagonal_4()))
    @test diagonal_graph(2; A=B, B=A) == diagonal_2(; A=B, B=A)
    for width in (2, 4, 8, 16)
        for g in (row_graph(width), diagonal_graph(width))
            @test numout(g) == numin(g) == width
            @test g isa TubeGraph
            @test isplanar(open_graph(g))
            @test length(occurrences(closed_graph(g))) == width
            @test dualflags(closed_graph(g)) == (ntuple(_ -> false, width)..., ntuple(_ -> true, width)...)
        end
    end
    @test numout(shifted_row_8()) == 8
    @test length(occurrences(closed_graph(row_graph(8; layers=3)))) == 24
    for width in (-2, 0, 1, 3)
        @test_throws ArgumentError row_graph(width)
        @test_throws ArgumentError diagonal_graph(width)
    end
    @test_throws ArgumentError row_graph(2; layers=0)
    @test_throws ArgumentError row_graph(2; A=TensorNode(:C, (1, 1)))
end

@testset "Auxiliary graphs retain their orientation" begin
    A = TensorNode(:A, (2, 2))
    B = TensorNode(:B, (2, 2))
    p1 = ((3, 1), (4, 2))
    p2 = ((2, 4), (1, 3))
    horizontal = double_row_2(; A=permute(B, p1), B=permute(A, p1))
    @test horizontal ==
        substitute(double_row_2(), A=>permute(B, p1), B=>permute(A, p1))

    # Compose the four sequential operations in the reference action. A pair of
    # untouched state legs remains exposed at each intermediate boundary.
    X = TensorNode(:X, (4, 1))
    @graph s1[u1 u2 u3 u4; z] := B[u2 u3; a b] * X[u1 a b u4; z]
    @graph s2[v1 v2 v3 v4; z] := A[v1 v2; a b] * s1[a b v3 v4; z]
    @graph s3[w1 w2 w3 w4; z] := A[w3 w4; a b] * s2[w1 w2 a b; z]
    @graph reference[o1 o2 o3 o4; z] := B[o4 o1; a b] * s3[o3 a b o2; z]
    G = TensorLatticeLab.tilted_4()
    physical = closed_graph(G)
    @graph action[o1 o2 o3 o4; z] := physical[o1 o2 o3 o4; p1 p2 p3 p4] * X[p1 p2 p3 p4; z]
    @test action == reference
    antidiagonal = TensorLatticeLab.tilted_4(; A=permute(B, p2), B=permute(A, p2))
    @test antidiagonal ==
        substitute(G, A=>permute(B, p2), B=>permute(A, p2))
    for g in (double_row_2(), horizontal, G, antidiagonal)
        @test length(occurrences(expand(closed_graph(g)))) == 4
    end
end
