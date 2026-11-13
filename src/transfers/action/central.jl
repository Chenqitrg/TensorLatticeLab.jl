"""Direct sum of sector actions on a central tensor map, with one TransferIrrep per occupied bulk sector."""
struct TransferRepresentation{S<:CentralTensorMapSpace,P,T}
    space::S
    plans::P
    scalar::Type{T}
end

# B distinguishes physical leg spaces even when they share a sector (e.g. ℝ and ℂ).
const CentralSpaceOn{B} = CentralTensorMapSpace{I,C,N,<:ProductSpace{B}} where {I<:Sector,C<:DrinfeldCenter{I},N}

function prepare_action(op::TransferOperator{E,B}, S::CentralSpaceOn{B}) where {E,B}
    codomain(S) == domain(op) || throw(SpaceMismatch("central state has the wrong physical legs"))
    plans = Dict(a => (prepare_action(op.tube, op.coloring, a),
                       domain(op) ← transfer_charge_space(spacetype(op), a)) for a in blocksectors(S))
    # A half-braiding can be complex even when the lattice gates are real.
    T = scalartype(op)
    for (irrep, _) in values(plans)
        for h in get(irrep.coloring.tensors, :gamma, ())
            T = promote_type(T, scalartype(h))
        end
    end
    return TransferRepresentation(S, plans, T)
end
prepare_action(op::SectorTransferOperator{I}, V::GradedSpace{C}) where {I<:Sector,C<:DrinfeldCenter{I}} =
    prepare_action(op, CentralTensorMapSpace(domain(op), V))
prepare_action(op::SectorTransferOperator{I}, a::C) where {I<:Sector,C<:DrinfeldCenter{I}} =
    prepare_action(op, Vect[C](a=>1))
# B is the elementary space of the central map's physical output legs.
const CentralMapOn{B} = CentralTensorMap{T,A,S} where {T,A,S<:CentralSpaceOn{B}}

prepare_action(op::TransferOperator{E,B}, U::CentralMapOn{B}) where {E,B} = prepare_action(op, space(U))
(op::TransferOperator{E,B})(U::CentralMapOn{B}) where {E,B} = prepare_action(op, U)(U)

# Retain the boundary-state entry point for a prepared single-sector contraction.
function (action::TransferRepresentation)(ψ::AbstractTensorMap)
    irrep, W = only(values(action.plans))
    space(ψ) == W || throw(SpaceMismatch("incompatible boundary state"))
    return irrep(ψ)
end

function (action::TransferRepresentation)(U::CentralTensorMap)
    space(U) == action.space || throw(SpaceMismatch("central state differs from the prepared space"))
    if length(action.plans) == 1
        a, (irrep, W) = only(action.plans)
        if size(block(U,a),2) == 1
            ψ = TensorMap{eltype(U)}(U.data, W)
            return CentralTensorMap(irrep(ψ).data, action.space)
        end
    end
    result = CentralTensorMap{promote_type(eltype(U), action.scalar)}(undef, action.space)
    for a in blocksectors(U)
        irrep, W = action.plans[a]
        input, output = block(U,a), block(result,a)
        for j in axes(input,2)
            ψ = TensorMap{eltype(U)}(input[:,j], W)
            output[:,j] .= irrep(ψ).data
        end
    end
    return result
end
