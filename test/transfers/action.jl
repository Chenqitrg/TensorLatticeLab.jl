import BoundaryBulkKit

# A deliberately tiny state and gates: no full transfer tensor is formed.
@testset "Prepared transfer action" begin
    V = ℂ^2
    X = ℂ^1
    A = randn(ComplexF64, V ⊗ V ← V ⊗ V)
    B = randn(ComplexF64, V ⊗ V ← V ⊗ V)
    g = diagonal_4()
    op = TransferOperator(g, checkerboard_coloring(g, (A, B)))
    action = anyon_insertion_irrep(one(BoundaryBulkKit.take_center(Trivial)), op)
    ψ = randn(ComplexF64, domain(op) ← X)
    @tensor expected[o1 o2 o3 o4; x] :=
        (((ψ[p1 p2 p3 p4; x] * B[i1 i2; p1 p2]) *
          B[i3 i4; p3 p4]) * A[o2 o3; i2 i3]) * A[o4 o1; i4 i1]
    G = action_graph(g, TensorNode(:ψ, ψ))
    @test isplanar(G)
    @test count("gamma", sprint(showgraph, G)) == length(g.annular)
    c = TensorGraphLab.Coloring(merge(action.coloring.tensors, (; ψ)))
    @test evaluate(G, c; method=:planar, small_indices=(:x0,:x1,:x2)) ≈ expected
    @test action(ψ) ≈ expected
    @test action(2ψ) ≈ 2expected
    @test_throws SpaceMismatch action(randn(ComplexF64, V ⊗ V ← X))
    @test_throws SpaceMismatch action(randn(ComplexF64, domain(op) ← V))
end

@testset "Graph families and arbitrary local rank" begin
    V, X = ℂ^1, ℂ^1
    a = ones(ComplexF64, V ⊗ V ← V ⊗ V)
    anyon = one(BoundaryBulkKit.take_center(Trivial))
    for graph in (diagonal_2(), diagonal_4(), shifted_row_2(), double_row_2(),
                  row_4(), shifted_row_4(), shifted_row_8(), TensorLatticeLab.tilted_4(),
                  row_graph(6; layers=3, shift=-1), diagonal_graph(6; shift=2))
        op = TransferOperator(graph, checkerboard_coloring(graph, (a, a)))
        action = anyon_insertion_irrep(anyon, op)
        ψ = ones(ComplexF64, domain(op) ← X)
        @test action(ψ) ≈ ψ
    end
    V = ℂ^2
    a = randn(ComplexF64, V ⊗ V ⊗ V ← V ⊗ V ⊗ V)
    A = TensorNode(:A, a)
    @tube annular=() graph[o1 o2 o3; i1 i2 i3] := A[o1 o2 o3; i1 i2 i3]
    op = TransferOperator(graph, coloring(graph, A => a))
    ψ = randn(ComplexF64, domain(op) ← X)
    @test anyon_insertion_irrep(anyon, op)(ψ) ≈ a * ψ
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
            action = anyon_insertion_irrep(anyon, op)
            @test action(ψ) ≈ reference_diagonal_action(A, B, ψ, anyon)
            @test anyon_insertion_irrep(anyon, squareop)(ψ) ≈ action(action(ψ))
            @test dot(φ, action(ψ)) ≈ dot(anyon_insertion_irrep(anyon, adjop)(φ), ψ)
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
    anyon = one(BoundaryBulkKit.take_center(Trivial))
    action = anyon_insertion_irrep(anyon, op)
    hv, hw = BraidingTensor(X,V)', BraidingTensor(X,W)'
    @planar intermediate[o;x] := (a[v o; i w] * ψ[i;y]) * hv[y w;v x]
    @planar expected[o;x] := (b[v o; i w] * intermediate[i;y]) * hw[y w;v x]
    @test action(ψ) ≈ expected
end

include("action_references.jl")

@testset "Curried transfer operator insertion" begin
    V = ℂ^2
    a = randn(ComplexF64, V ⊗ V ← V ⊗ V)
    b = randn(ComplexF64, V ⊗ V ← V ⊗ V)
    tube = diagonal_2()
    lattice = checkerboard_coloring(tube, (a,b))
    op = TransferOperator(tube, lattice)
    G = action_graph(tube)
    C = BoundaryBulkKit.take_center(Trivial)
    charge = one(C)
    irrep = anyon_insertion_irrep(charge, op)
    ψ = randn(ComplexF64, irrep.space)
    bindings = TensorGraphLab.Coloring(merge(irrep.coloring.tensors, (; ψ)))
    @test irrep.graph == G
    reference = evaluate(irrep.graph, bindings; method=:planar, opt=true)
    @test irrep(ψ) ≈ reference
    @test anyon_insertion_irrep(charge)(op)(ψ) ≈ reference
    @test !applicable(anyon_insertion_irrep(charge), G, lattice)
    @test irrep.coloring[:A] === lattice[:A]
    @test irrep.coloring[:B] === lattice[:B]
    bulk = Vect[C](charge => 2)
    rep = anyon_insertion_rep(bulk, op)
    U = BoundaryBulkKit.CentralTensorMap(vcat(ψ.data, 2ψ.data), codomain(ψ), bulk)
    expected = vcat(reference.data, 2reference.data)
    @test rep(U).data ≈ expected
    @test anyon_insertion_rep(bulk)(op)(U).data ≈ expected
    @test !applicable(anyon_insertion_rep(bulk), G, lattice)
    @test_throws SpaceMismatch rep(BoundaryBulkKit.CentralTensorMap(copy(ψ.data), codomain(ψ), Vect[C](charge=>1)))
    empty_rep = anyon_insertion_rep(Vect[C](), op)
    empty_state = zeros(ComplexF64, empty_rep.space)
    @test empty_rep(empty_state) ≈ empty_state
end
