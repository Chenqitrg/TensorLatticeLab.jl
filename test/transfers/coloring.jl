@testset "Default A/B coloring without contraction" begin
    # One-dimensional fixtures test binding and spaces only. Never evaluate a
    # transfer tensor here: its size scales exponentially with boundary width.
    V = ℂ^1
    a = TensorMap(ones(ComplexF64, 1, 1), V ⊗ V ← V ⊗ V)
    b = 2a
    g = diagonal_4()
    c = coloring(g, (a, b))
    @test c[:A] === a
    @test c[:B] === b
    c = coloring(g, a)
    @test c[:A] === c[:B] === a
    A = TensorNode(:A, a)
    @graph single[o1 o2; p1 p2] := A[o1 o2; p1 p2]
    @test keys(coloring(single, (a, b)).tensors) == (:A,)
    C = TensorNode(:C, a)
    @graph custom[o1 o2; p1 p2] := C[o1 o2; p1 p2]
    @test_throws ArgumentError coloring(custom, a)
    @test_throws MethodError coloring(g, (a, b, a, b))
    @test coloring(custom, :C=>a)[:C] === a
    B = TensorNode(:B, b)
    p1 = ((3, 1), (4, 2))
    p2 = ((2, 4), (1, 3))
    for h in (double_row_2(), TensorLatticeLab.tilted_4(),
              double_row_2(; A=permute(B, p1), B=permute(A, p1)),
              TensorLatticeLab.tilted_4(; A=permute(B, p2), B=permute(A, p2)))
        @test isnothing(TensorGraphLab.validate(h, coloring(h, (a, b))))
    end
end

using Test, TensorKit, TensorGraphLab, TensorLatticeLab

@testset "Tensor-cell coloring" begin
    V = ℂ^1
    a = TensorMap(ones(ComplexF64, 1, 1), V ⊗ V ← V ⊗ V)
    g = diagonal_4()
    uniform = TensorLatticeLab.UniformGates(a)
    c = coloring(g, uniform)
    @test c[:A] === c[:B] === uniform.gate
    checkerboard = TensorLatticeLab.CheckerboardGates(a, 2a)
    c = coloring(g, checkerboard)
    @test c[:A] === checkerboard.black
    @test c[:B] === checkerboard.white
    plaquette = TensorLatticeLab.PlaquetteGates(a, a, a, a)
    @test_throws MethodError coloring(g, plaquette)
end
