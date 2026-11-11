# Independent state-first references. Use local gates and physical translations
# directly; the references do not inspect a TubeGraph or call action_graph.
import BoundaryBulkKit
using Random: Xoshiro

function translate_right(ψ::AbstractTensorMap{E,S,2,1}, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2; x] := ψ[o2 v; y] * h[o1 y; x v]
    return result
end

function translate_left(ψ::AbstractTensorMap{E,S,2,1}, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2; x] := ψ[v o1; y] * h'[y o2; v x]
    return result
end

function reference_row(ψ::AbstractTensorMap{E,S,2,1}, A::AbstractTensorMap, B::AbstractTensorMap, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2; x] :=
        ψ[i1 i2; y] * A[b0 o1; i1 b1] *
        B[b1 o2; i2 b2] *
        h'[y b2; b0 x]
    return result
end

function translate_right(ψ::AbstractTensorMap{E,S,4,1}, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2 o3 o4; x] := ψ[o2 o3 o4 v; y] * h[o1 y; x v]
    return result
end

function translate_left(ψ::AbstractTensorMap{E,S,4,1}, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2 o3 o4; x] := ψ[v o1 o2 o3; y] * h'[y o4; v x]
    return result
end

function reference_row(ψ::AbstractTensorMap{E,S,4,1}, A::AbstractTensorMap, B::AbstractTensorMap, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2 o3 o4; x] :=
        ψ[i1 i2 i3 i4; y] * A[b0 o1; i1 b1] *
        B[b1 o2; i2 b2] * A[b2 o3; i3 b3] *
        B[b3 o4; i4 b4] *
        h'[y b4; b0 x]
    return result
end

function translate_right(ψ::AbstractTensorMap{E,S,6,1}, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2 o3 o4 o5 o6; x] := ψ[o2 o3 o4 o5 o6 v; y] * h[o1 y; x v]
    return result
end

function translate_left(ψ::AbstractTensorMap{E,S,6,1}, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2 o3 o4 o5 o6; x] := ψ[v o1 o2 o3 o4 o5; y] * h'[y o6; v x]
    return result
end

function reference_row(ψ::AbstractTensorMap{E,S,6,1}, A::AbstractTensorMap, B::AbstractTensorMap, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2 o3 o4 o5 o6; x] :=
        ψ[i1 i2 i3 i4 i5 i6; y] * A[b0 o1; i1 b1] *
        B[b1 o2; i2 b2] * A[b2 o3; i3 b3] *
        B[b3 o4; i4 b4] * A[b4 o5; i5 b5] *
        B[b5 o6; i6 b6] *
        h'[y b6; b0 x]
    return result
end

function translate_right(ψ::AbstractTensorMap{E,S,8,1}, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2 o3 o4 o5 o6 o7 o8; x] := ψ[o2 o3 o4 o5 o6 o7 o8 v; y] * h[o1 y; x v]
    return result
end

function reference_row(ψ::AbstractTensorMap{E,S,8,1}, A::AbstractTensorMap, B::AbstractTensorMap, h::AbstractTensorMap) where {E,S}
    @planar result[o1 o2 o3 o4 o5 o6 o7 o8; x] :=
        ψ[i1 i2 i3 i4 i5 i6 i7 i8; y] * A[b0 o1; i1 b1] *
        B[b1 o2; i2 b2] * A[b2 o3; i3 b3] *
        B[b3 o4; i4 b4] * A[b4 o5; i5 b5] *
        B[b5 o6; i6 b6] * A[b6 o7; i7 b7] *
        B[b7 o8; i8 b8] *
        h'[y b8; b0 x]
    return result
end

function reference_diagonal_2(ψ::AbstractTensorMap, A::AbstractTensorMap, B::AbstractTensorMap, h::AbstractTensorMap)
    return translate_right(A * translate_left(B * ψ, h), h)
end

function reference_diagonal_6(ψ::AbstractTensorMap, A::AbstractTensorMap, B::AbstractTensorMap, h::AbstractTensorMap)
    @planar lower[a b c d e f; x] :=
        ((ψ[i j k l m n; x] * B[a b; i j]) * B[c d; k l]) * B[e f; m n]
    shifted = translate_left(lower, h)
    @planar upper[a b c d e f; x] :=
        ((shifted[i j k l m n; x] * A[a b; i j]) * A[c d; k l]) * A[e f; m n]
    return translate_right(upper, h)
end

function reference_tilted_4(ψ::AbstractTensorMap, A::AbstractTensorMap, B::AbstractTensorMap, h::AbstractTensorMap)
    # Four local operations in TNRKit's NE-to-SW order. Their natural output
    # order is (o3,o4,o1,o2); translate left twice to reach the tube boundary.
    @planar s1[a b c d; x] := ψ[a i j d; x] * B[b c; i j]
    @planar s2[a b c d; x] := s1[i j c d; x] * A[a b; i j]
    @planar s3[a b c d; x] := s2[a b i j; x] * A[c d; i j]
    @planar s4[a b c d; x] := s3[a i j d; x] * B[b c; i j]
    return translate_left(translate_left(s4, h), h)
end

function check_reference(name::String, tube::TubeGraph, A::AbstractTensorMap, B::AbstractTensorMap,
                         ψ::AbstractTensorMap, anyon::Sector, expected::AbstractTensorMap)
    op = TransferOperator(tube, checkerboard_coloring(tube, (A, B)))
    actual = prepare_action(op, anyon)(ψ)
    @testset "$name, $anyon" begin
        @test space(actual) == space(expected)
        @test norm(expected) > 0
        residual = norm(actual - expected) / norm(expected)
        @test residual < 1e-11
        @debug "Reference action residual" name anyon residual
        if anyon isa BoundaryBulkKit.Z3QuantumDouble && anyon.charge == 1 && anyon.flux == 1
            # Negating the flux conjugates the crossing phases without changing X.
            reverse_flux = typeof(anyon)(anyon.charge, -anyon.flux)
            wrong = prepare_action(op, reverse_flux)(ψ)
            reverse_residual = norm(wrong - expected) / norm(expected)
            @test reverse_residual > 1e-3
            @debug "Reversed crossing phase residual" name reverse_residual
        end
    end
end

@testset "Independent small-bond actions on each transfer geometry" begin
    rng = Xoshiro(0xace29)
    for S in (Z3Irrep, FibonacciAnyon), width in (2, 4, 6, 8)
        # Each local sector has multiplicity one. A single Fibonacci τ on
        # the wider rows still has multiple fusion paths and nontrivial braiding.
        V = if S === Z3Irrep
            Z3Space(0=>1, 1=>1)
        elseif width <= 4
            Vect[S](:I=>1, :τ=>1)
        else
            Vect[S](:τ=>1)
        end
        A = randn(rng, ComplexF64, V ⊗ V ← V ⊗ V)
        B = randn(rng, ComplexF64, V ⊗ V ← V ⊗ V)
        physical = ProductSpace(ntuple(_ -> V, width))
        for anyon in values(BoundaryBulkKit.take_center(S))
            state_space = physical ← BoundaryBulkKit.forget(anyon)
            @test 0 < dim(state_space) <= 86
            ψ = randn(rng, ComplexF64, state_space)
            h = BoundaryBulkKit.HalfBraiding(anyon, V)
            row = reference_row(ψ, A, B, h)
            if width == 2
                check_reference("diagonal_2", diagonal_2(), A, B, ψ, anyon, reference_diagonal_2(ψ, A, B, h))
                # The +1/2 time period translates the output one site to the left.
                # The other lift has the same permutation and differs by a full twist.
                left = translate_left(row, h)
                right = translate_right(row, h)
                @test right ≈ twist(anyon) * left
                check_reference("shifted_row_2", shifted_row_2(), A, B, ψ, anyon, left)
                check_reference("double_row_2", double_row_2(), A, B, ψ, anyon, reference_row(row, B, A, h))
            elseif width == 4
                check_reference("diagonal_4", diagonal_4(), A, B, ψ, anyon, reference_diagonal_action(A, B, ψ, anyon))
                check_reference("row_4", row_4(), A, B, ψ, anyon, row)
                check_reference("shifted_row_4", shifted_row_4(), A, B, ψ, anyon, translate_right(row, h))
                check_reference("tilted_4", TensorLatticeLab.tilted_4(), A, B, ψ, anyon, reference_tilted_4(ψ, A, B, h))
            elseif width == 6
                rows = reference_row(reference_row(row, B, A, h), A, B, h)
                check_reference("row(6,3,-1)", row_graph(6; layers=3, shift=-1), A, B, ψ, anyon, translate_left(rows, h))
                diagonal = reference_diagonal_6(ψ, A, B, h)
                check_reference("diagonal(6,+2)", diagonal_graph(6; shift=2), A, B, ψ, anyon,
                                translate_right(translate_right(diagonal, h), h))
            else
                check_reference("shifted_row_8", shifted_row_8(), A, B, ψ, anyon, translate_right(row, h))
            end
        end
    end
end
