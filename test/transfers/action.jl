import BoundaryBulkKit

# A deliberately tiny state and gates: no full transfer tensor is formed.
@testset "Prepared transfer action" begin
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
    op = TransferOperator(g, checkerboard_coloring(g, (A, B)))
    action = prepare_action(op, X, halfbraiding)
    @test fieldnames(typeof(action)) == (:graph, :coloring, :plan)
    prepared_calls = calls[]
    @test prepared_calls == 2  # V and its dual cross the diagonal seam.
    ψ = randn(ComplexF64, domain(op) ← X)
    @tensor expected[o1 o2 o3 o4; x] :=
        (((ψ[p1 p2 p3 p4; x] * B[i1 i2; p1 p2]) *
          B[i3 i4; p3 p4]) * A[o2 o3; i2 i3]) * A[o4 o1; i4 i1]
    G = action_graph(g, TensorNode(:ψ, ψ))
    @test isplanar(G)
    @test count("gamma", sprint(showgraph, G)) == length(g.annular)
    @test G == action.graph
    c = coloring(action, ψ)
    @test evaluate(G, c; method=:planar, small_indices=(:x0,:x1,:x2)) ≈ expected
    @test action(ψ) ≈ expected
    @test action(2ψ) ≈ 2expected
    @test calls[] == prepared_calls
    @test_throws SpaceMismatch action(randn(ComplexF64, V ⊗ V ← X))
    @test_throws SpaceMismatch prepare_action(op, X, W -> id(W ⊗ X))
end

@testset "Graph families and arbitrary local rank" begin
    V, X = ℂ^1, ℂ^1
    a = ones(ComplexF64, V ⊗ V ← V ⊗ V)
    gamma = W -> copy(BraidingTensor(X, W))
    for graph in (diagonal_2(), diagonal_4(), shifted_row_2(), double_row_2(),
                  row_4(), shifted_row_4(), shifted_row_8(), TensorLatticeLab.tilted_4(),
                  row_graph(6; layers=3, shift=-1), diagonal_graph(6; shift=2))
        op = TransferOperator(graph, checkerboard_coloring(graph, (a, a)))
        action = prepare_action(op, X, gamma)
        ψ = ones(ComplexF64, domain(op) ← X)
        @test action(ψ) ≈ ψ
    end
    V = ℂ^2
    a = randn(ComplexF64, V ⊗ V ⊗ V ← V ⊗ V ⊗ V)
    A = TensorNode(:A, a)
    @tube annular=() graph[o1 o2 o3; i1 i2 i3] := A[o1 o2 o3; i1 i2 i3]
    op = TransferOperator(graph, coloring(graph, A => a))
    ψ = randn(ComplexF64, domain(op) ← X)
    @test prepare_action(op, X, gamma)(ψ) ≈ a * ψ
end

# An independent planar brickwork action: move the first output to the end,
# apply the staggered gates, then move it back using the positive half-braiding.
function reference_diagonal_action(A::AbstractTensorMap, B::AbstractTensorMap,
                                   ψ::AbstractTensorMap, anyon::Sector)
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
        op = TransferOperator(g, checkerboard_coloring(g, (A, B)))
        # Stack the actual open patches so that both layers retain their seams.
        P = open_graph(g)
        @graph layers[l1 l2 l3 l4 o1 o2 o3 o4; i1 i2 i3 i4 r1 r2 r3 r4] :=
            P[l3 l4 o1 o2 o3 o4; j1 j2 j3 j4 r3 r4] * P[l1 l2 j1 j2 j3 j4; i1 i2 i3 i4 r1 r2]
        squared = TubeGraph((:a,:b,:c,:d), layers)
        squareop = TransferOperator(squared, checkerboard_coloring(squared, (A, B)))
        @graph reflected[l2 l1 i1 i2 i3 i4; o1 o2 o3 o4 r2 r1] :=
            P'[i1 i2 i3 i4 r1 r2; l1 l2 o1 o2 o3 o4]
        adj = TubeGraph((:b,:a), reflected)
        adjop = TransferOperator(adj, coloring(adj, :A=>A, :B=>B))
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

@testset "One gamma family on unequal annular spaces" begin
    X, U, V, W = ℂ^1, ℂ^2, ℂ^2, ℂ^3
    a = randn(ComplexF64, V ⊗ U ← U ⊗ V)
    b = randn(ComplexF64, W ⊗ U ← U ⊗ W)
    A, B = TensorNode(:A, a), TensorNode(:B, b)
    @tube annular=(s,t) tube[o;i] := B[t o; m t] * A[s m; i s]
    op = TransferOperator(tube, coloring(tube, A=>a, B=>b))
    ψ = randn(ComplexF64, U ← X)
    observed = Set{ComplexSpace}()
    function gamma(Y::ElementarySpace)
        push!(observed, Y)
        return copy(BraidingTensor(X, Y))
    end
    action = prepare_action(op, X, gamma)
    @test observed == Set((V,W))
    hv, hw = gamma(V)', gamma(W)'
    @planar intermediate[o;x] := (a[v o; i w] * ψ[i;y]) * hv[y w;v x]
    @planar expected[o;x] := (b[v o; i w] * intermediate[i;y]) * hw[y w;v x]
    @test action(ψ) ≈ expected
end

include("action_references.jl")

@testset "Prepare from an action graph, marked tube or transfer operator" begin
    V, X = ℂ^2, ℂ^1
    a = randn(ComplexF64, V ⊗ V ← V ⊗ V)
    b = randn(ComplexF64, V ⊗ V ← V ⊗ V)
    tube = diagonal_2()
    lattice = checkerboard_coloring(tube, (a,b))
    op = TransferOperator(tube, lattice)
    ψ = randn(ComplexF64, domain(op) ← X)
    G = action_graph(tube; state_name=:state)
    gamma(W::ElementarySpace) = copy(BraidingTensor(X,W))
    by_space = Dict(W => gamma(W) for W in (V, dual(V)))
    reference = prepare_action(op, X, gamma)(ψ)
    direct = prepare_action(G, lattice; gamma=by_space, state_name=:state)
    @test direct.coloring[:A] === lattice[:A]
    @test direct.coloring[:B] === lattice[:B]
    @test Set(keys(direct.coloring.tensors)) == Set((:A,:B,:gamma))
    @test prepare_action(tube, lattice; gamma)(ψ) ≈ reference
    @test prepare_action(op; gamma)(ψ) ≈ reference
    @test direct(ψ) ≈ reference
    @test direct(2ψ) ≈ 2reference
    @test_throws SpaceMismatch coloring(direct, randn(ComplexF64, domain(op) ← V))
    completed = coloring(direct, ψ)
    @test completed[:state] === ψ
    @test evaluate(G, completed; method=:planar, opt=true) ≈ direct(ψ)
    @test prepare_action(G, completed; state_name=:state)(2ψ) ≈ 2reference
    @test prepare_action(G, (; A=a, B=b, gamma); state_name=:state)(ψ) ≈ reference
    @test prepare_action(tube, (; A=a, B=b, gamma=by_space))(ψ) ≈ reference
    @test prepare_action(G, (; A=a, B=b, gamma=direct.coloring[:gamma]); state_name=:state)(ψ) ≈ reference
    @test prepare_action(tube, lattice, X, gamma)(ψ) ≈ reference
    @test_throws SpaceMismatch prepare_action(G, completed; state_name=:state, state_space=V ← X)
    @test_throws ArgumentError prepare_action(G, lattice; gamma=(gamma(V),), state_name=:state)

    # A compiled plan resolves the state by its color even if it occurs first.
    terms = TensorGraphLab.occurrences(G)
    reordered = TensorGraph((terms[end], terms[1:end-1]...), TensorGraphLab.boundary(G)...)
    @test prepare_action(reordered, completed; state_name=:state)(ψ) ≈ reference

    tube = row_4()
    lattice = checkerboard_coloring(tube, (a,b))
    four_legs = randn(ComplexF64, domain(space(tube,lattice)) ← X)
    h = gamma(V)
    @test prepare_action(action_graph(tube), lattice; gamma=h)(four_legs) ≈
          prepare_action(tube, lattice, X, gamma)(four_legs)
end
