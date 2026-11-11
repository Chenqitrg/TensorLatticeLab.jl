# A deliberately tiny state and gates: no full transfer tensor is formed.
@testset "Prepared transfer action" begin
    @test isdefined(TensorLatticeLab, :prepare_action)
    if isdefined(TensorLatticeLab, :prepare_action)
        V = ℂ^2
        X = ℂ^1
        A = randn(ComplexF64, V ⊗ V ← V ⊗ V)
        B = randn(ComplexF64, V ⊗ V ← V ⊗ V)
        calls = Ref(0)
        halfbraiding = W -> begin
            calls[] += 1
            copy(BraidingTensor(X, W))
        end
        g = diagonal_4()
        op = TransferOperator(g, coloring(g, (A, B)))
        action = prepare_action(op, X, halfbraiding)
        prepared_calls = calls[]
        ψ = randn(ComplexF64, domain(op) ← X)
        @tensor expected[o1 o2 o3 o4; x] :=
            (((ψ[p1 p2 p3 p4; x] * B[i1 i2; p1 p2]) *
              B[i3 i4; p3 p4]) * A[o2 o3; i2 i3]) * A[o4 o1; i4 i1]
        @test action(ψ) ≈ expected
        @test action(2ψ) ≈ 2expected
        @test calls[] == prepared_calls
        @test_throws SpaceMismatch action(randn(ComplexF64, V ⊗ V ← X))
        @test_throws SpaceMismatch prepare_action(op, X, W -> id(W ⊗ X))
    end
end

@testset "Graph families and arbitrary local rank" begin
    V, X = ℂ^1, ℂ^1
    a = ones(ComplexF64, V ⊗ V ← V ⊗ V)
    gamma = W -> copy(BraidingTensor(X, W))
    for graph in (diagonal_2(), diagonal_4(), shifted_row_2(), double_row_2(),
                  row_4(), shifted_row_4(), shifted_row_8(), TensorLatticeLab.tilted_4(),
                  row_graph(6; layers=3, shift=-1), diagonal_graph(6; shift=2))
        op = TransferOperator(graph, coloring(graph, (a, a)))
        action = prepare_action(op, X, gamma)
        ψ = ones(ComplexF64, domain(op) ← X)
        @test action(ψ) ≈ ψ
    end
    V = ℂ^2
    a = randn(ComplexF64, V ⊗ V ⊗ V ← V ⊗ V ⊗ V)
    A = TensorNode(:A, a)
    @graph graph[o1 o2 o3; i1 i2 i3] := A[o1 o2 o3; i1 i2 i3]
    op = TransferOperator(graph, coloring(graph, A => a))
    ψ = randn(ComplexF64, domain(op) ← X)
    @test prepare_action(op, X, gamma)(ψ) ≈ a * ψ

    g = diagonal_4()
    @graph crossed[o1 o2 o3 o4; i1 i2 i3 i4] := g[o1 o3 o2 o4; i1 i2 i3 i4]
    t = ones(ComplexF64, ℂ^1 ⊗ ℂ^1 ← ℂ^1 ⊗ ℂ^1)
    @test_throws ArgumentError prepare_action(TransferOperator(crossed, coloring(crossed, t)), X, gamma)
end

# An independent planar brickwork action: move the first output to the end,
# apply the staggered gates, then move it back using the positive half-braiding.
function reference_diagonal_action(A, B, ψ, anyon)
    V = domain(A, 1)
    h = BoundaryBulkKit.HalfBraiding(anyon, V)
    hi = copy(h')
    @planar lower[a b c d; x] := (ψ[i j k l; x] * B[a b; i j]) * B[c d; k l]
    @planar shifted[b c d a; x] := lower[i b c d; y] * hi[y a; i x]
    @planar upper[b c d a; x] := (shifted[i j k l; x] * A[b c; i j]) * A[d a; k l]
    @planar result[a b c d; x] := upper[b c d i; y] * h[a y; x i]
    return result
end

@testset "Center sectors and functorial action" begin
    for S in (Z3Irrep, FibonacciAnyon)
        V = S === Z3Irrep ? Z3Space(0=>1, 1=>1, 2=>1) : Vect[S](one(S)=>1, FibonacciAnyon(:τ)=>1)
        A = randn(ComplexF64, V ⊗ V ← V ⊗ V)
        B = randn(ComplexF64, V ⊗ V ← V ⊗ V)
        g = diagonal_4()
        op = TransferOperator(g, coloring(g, (A, B)))
        adjop = TransferOperator(g', coloring(g', (A, B)))
        @graph squared[o1 o2 o3 o4; i1 i2 i3 i4] :=
            g[o1 o2 o3 o4; j1 j2 j3 j4] * g[j1 j2 j3 j4; i1 i2 i3 i4]
        squareop = TransferOperator(squared, coloring(squared, (A, B)))
        for anyon in values(BoundaryBulkKit.take_center(S))
            X = BoundaryBulkKit.forget(anyon)
            ψ = randn(ComplexF64, domain(op) ← X)
            φ = randn(ComplexF64, domain(op) ← X)
            action = prepare_action(op, anyon)
            @test action(ψ) ≈ reference_diagonal_action(A, B, ψ, anyon)
            @test prepare_action(squareop, anyon)(ψ) ≈ action(action(ψ))
            @test dot(φ, action(ψ)) ≈ dot(prepare_action(adjop, anyon)(φ), ψ)
        end
    end
end
