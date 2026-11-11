# A bounded diagnostic of the information retained by an ordinary TensorGraph.
# This does not expose a prepare_action API or materialize a transfer tensor.
include(joinpath(@__DIR__, "..", "..", "examples", "_setup.jl"))
using Test, TensorKit, TensorGraphLab, TensorLatticeLab
import BoundaryBulkKit
import Random

function rotate_state_once(psi, inverse_halfbraiding)
    @planar result[b a; x] := psi[i b; y] * inverse_halfbraiding[y a; i x]
    return result
end

@testset "Flattened connectivity does not retain a translation lift" begin
    I = TensorNode(:I, (1, 1))
    @graph shift[o1 o2; i1 i2] := I[o1; i2] * I[o2; i1]
    @graph twice[o1 o2; i1 i2] := shift[o1 o2; j1 j2] * shift[j1 j2; i1 i2]
    @graph straight[o1 o2; i1 i2] :=
        I[o1; j1] * I[j1; i1] * I[o2; j2] * I[j2; i2]

    # The graphs have the same external leg order, number of vertices, colors,
    # local cyclic orders and connected paths, up to internal relabeling.
    @test twice == straight
    @test TensorGraphLab.expand(twice) == TensorGraphLab.expand(straight)

    V = Z3Space(0 => 1, 1 => 1, 2 => 1)
    anyon = BoundaryBulkKit.ZNQuantumDouble{3}(1, 1)
    X = BoundaryBulkKit.forget(anyon)
    Random.seed!(6417)
    psi = randn(ComplexF64, V ⊗ V ← X)
    hi = copy(BoundaryBulkKit.HalfBraiding(anyon, V)')
    rotated = rotate_state_once(rotate_state_once(psi, hi), hi)

    # Moving each leg once through the fixed n|1 seam gives the inverse twist
    # for this left-rotation convention, not the identity action.
    @test rotated ≈ conj(twist(anyon)) * psi
    @test !isapprox(rotated, psi)
    println("Two shifts and two straight identity layers have equal TensorGraphs.")
    println("Two charge-aware rotations / identity: ", dot(psi, rotated) / dot(psi, psi))
    println("Expected inverse twist: ", conj(twist(anyon)))
    println("Relative residual: ", norm(rotated - conj(twist(anyon)) * psi) / norm(psi))
end
