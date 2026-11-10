const GenericGateTensor = AbstractTensorMap{E,S,2,2} where {E,S}

"""
    AbstractTensorCell{G} <: AbstractMatrix{G}

Periodic matrix unit cell of square four-leg gate tensors. Integer row and
column indices wrap around the stored unit cell. Construction standardizes
every periodic bond to a nondual space through paired gauge transformations.
"""
abstract type AbstractTensorCell{G<:GenericGateTensor} <: AbstractMatrix{G} end

Base.IndexStyle(::Type{<:AbstractTensorCell}) = IndexCartesian()
Base.checkbounds(
    ::Type{Bool}, ::AbstractTensorCell{G}, ::Integer, ::Integer
) where {G} = true

"""
    UniformGates(gate)

One-gate periodic 1×1 unit cell. Every lattice coordinate returns `gate`.
"""
struct UniformGates{G<:GenericGateTensor} <: AbstractTensorCell{G}
    gate::G
    function UniformGates(gate::G) where {G<:GenericGateTensor}
        raw = new{G}(gate)
        _assert_closed_cell_bonds(raw)
        gates = _standardize_cell_gates(raw)
        cell = new{G}(gates[1, 1])
        assertconsistency(cell)
        return cell
    end
end

Base.size(::UniformGates) = (1, 1)
Base.getindex(cell::UniformGates, ::Integer, ::Integer) = cell.gate
Base.copy(cell::UniformGates) = UniformGates(copy(cell.gate))

"""
    CheckerboardGates(black, white)
    CheckerboardGates(cell::UniformGates; copy=true)

Two-gate checkerboard unit cell. The stored row is `[black white]`; neighboring
rows are staggered as `[white black]`.
"""
struct CheckerboardGates{G<:GenericGateTensor} <: AbstractTensorCell{G}
    black::G
    white::G
    function CheckerboardGates(black::G, white::G) where {G<:GenericGateTensor}
        raw = new{G}(black, white)
        _assert_closed_cell_bonds(raw)
        gates = _standardize_cell_gates(raw)
        cell = new{G}(gates[1, 1], gates[1, 2])
        assertconsistency(cell)
        return cell
    end
end

Base.size(::CheckerboardGates) = (1, 2)
function Base.getindex(cell::CheckerboardGates, row::Integer, column::Integer)
    return iseven(row + column) ? cell.black : cell.white
end

_copy_gate(gate, copy::Bool) = copy ? Base.copy(gate) : gate

CheckerboardGates(cell::UniformGates; copy::Bool = true) = CheckerboardGates(
    _copy_gate(cell.gate, copy), _copy_gate(cell.gate, copy)
)

Base.copy(cell::CheckerboardGates) = CheckerboardGates(
    copy(cell.black), copy(cell.white)
)

"""
    PlaquetteGates(gates::AbstractMatrix; copy=true)
    PlaquetteGates(northwest, northeast, southwest, southeast)
    PlaquetteGates(cell::UniformGates; copy=true)
    PlaquetteGates(cell::CheckerboardGates; copy=true)
    PlaquetteGates(cell::PlaquetteGates; copy=true)

Arbitrary periodic 2×2 gate plaquette. Matrix coordinates follow Julia's
`[row, column]` convention for a periodically repeated unit cell.
"""
struct PlaquetteGates{G<:GenericGateTensor} <: AbstractTensorCell{G}
    northwest::G
    northeast::G
    southwest::G
    southeast::G
    function PlaquetteGates(
            northwest::G,
            northeast::G,
            southwest::G,
            southeast::G,
        ) where {G<:GenericGateTensor}
        raw = new{G}(northwest, northeast, southwest, southeast)
        _assert_closed_cell_bonds(raw)
        gates = _standardize_cell_gates(raw)
        cell = new{G}(gates[1, 1], gates[1, 2], gates[2, 1], gates[2, 2])
        assertconsistency(cell)
        return cell
    end
end

function PlaquetteGates(
        gates::AbstractMatrix{G}; copy::Bool = true
    ) where {G<:GenericGateTensor}
    size(gates) == (2, 2) || throw(DimensionMismatch(
        "a gate plaquette must have size (2, 2), got $(size(gates))"
    ))
    return PlaquetteGates(
        _copy_gate(gates[1, 1], copy),
        _copy_gate(gates[1, 2], copy),
        _copy_gate(gates[2, 1], copy),
        _copy_gate(gates[2, 2], copy),
    )
end

PlaquetteGates(cell::UniformGates; copy::Bool = true) = PlaquetteGates(
    _copy_gate(cell.gate, copy),
    _copy_gate(cell.gate, copy),
    _copy_gate(cell.gate, copy),
    _copy_gate(cell.gate, copy),
)

PlaquetteGates(cell::CheckerboardGates; copy::Bool = true) = PlaquetteGates(
    _copy_gate(cell.black, copy),
    _copy_gate(cell.white, copy),
    _copy_gate(cell.white, copy),
    _copy_gate(cell.black, copy),
)

PlaquetteGates(cell::PlaquetteGates; copy::Bool = true) =
    copy ? Base.copy(cell) : cell

Base.copy(cell::PlaquetteGates) = PlaquetteGates(
    copy(cell.northwest),
    copy(cell.northeast),
    copy(cell.southwest),
    copy(cell.southeast),
)

Base.size(::PlaquetteGates) = (2, 2)
function Base.getindex(cell::PlaquetteGates, row::Integer, column::Integer)
    wrapped_row = mod1(row, 2)
    wrapped_column = mod1(column, 2)
    if wrapped_row == 1
        return wrapped_column == 1 ? cell.northwest : cell.northeast
    end
    return wrapped_column == 1 ? cell.southwest : cell.southeast
end

_stored_index(::UniformGates, ::Integer, ::Integer) = CartesianIndex(1, 1)
function _stored_index(::CheckerboardGates, row::Integer, column::Integer)
    return CartesianIndex(1, iseven(row + column) ? 1 : 2)
end
function _stored_index(::PlaquetteGates, row::Integer, column::Integer)
    return CartesianIndex(mod1(row, 2), mod1(column, 2))
end
