using Test
using CategoryData
using TensorLatticeLab
import TensorLatticeLab: assertconsistency, standardize
using Random
using TensorKit

function test_gate(; west=ℂ^2, south=ℂ^3, north=south, east=west)
    return randn(ComplexF64, west ⊗ south, north ⊗ east)
end

gate_spaces(gate) = (codomain(gate)..., domain(gate)...)
all_nondual(cell) = all(
    !isdual(space) for gate in cell for space in gate_spaces(gate)
)
