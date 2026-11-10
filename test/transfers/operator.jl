@testset "Static transfer operator" begin
    a = ones(Float64, ℂ^1 ⊗ ℂ^1 ← ℂ^1 ⊗ ℂ^1)
    b = complex(a)
    g = diagonal_4()
    c = coloring(g, (a, b))
    op = TransferOperator(g, c)
    @test op.graph === g
    @test op.coloring === c
    @test domain(op) == codomain(op) == ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1
    @test space(op) == (codomain(op) ← domain(op))
    @test (numout(op), numin(op), numind(op)) == (4, 4, 8)
    @test spacetype(op) === ComplexSpace
    @test sectortype(op) === Trivial
    @test scalartype(op) === ComplexF64
    @test lattice_geometry(op).space_period == lattice_geometry(g).space_period
    @test lattice_geometry(op).time_period == lattice_geometry(g).time_period
    @test !applicable(op, a)

    # Equal total dimensions do not imply equal ordered boundary spaces.
    t = ones(ComplexF64, ℂ^2 ⊗ ℂ^3 ← ℂ^3 ⊗ ℂ^2)
    A = TensorNode(:A, t)
    @graph unequal[w s; n e] := A[w s; n e]
    @test_throws SpaceMismatch TransferOperator(unequal, coloring(unequal, A => t))

    V = U1Space(0 => 1, 1 => 1)
    t = ones(ComplexF64, V ← V')
    A = TensorNode(:A, t)
    @graph dual_mismatch[w; e] := A[w; e]
    @test_throws SpaceMismatch TransferOperator(dual_mismatch, coloring(dual_mismatch, A => t))
    @test_throws ArgumentError TransferOperator(g, TensorGraphLab.Coloring((A = a,)))

    t = ones(ComplexF64, V ← V)
    A = TensorNode(:A, t)
    @graph symmetric[w; e] := A[w; e]
    symmetric_op = TransferOperator(symmetric, coloring(symmetric, A => t))
    @test sectortype(symmetric_op) === sectortype(V)
    @test domain(symmetric_op) == codomain(symmetric_op) == ProductSpace(V)

    @graph closed = A[i; i]
    closed_op = TransferOperator(closed, coloring(closed, A => t))
    @test numind(closed_op) == 0
    @test domain(closed_op) == codomain(closed_op) == ProductSpace{typeof(V)}(())
end
