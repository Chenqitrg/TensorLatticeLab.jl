"""Direct sum of the insertion irreps on a central tensor map space."""
struct TransferRepresentation{T<:Number,I<:Sector,C<:DrinfeldCenter{I},S<:CentralTensorMapSpace{I,C}}
    space::S
    irreps::Dict{C,TransferIrrep{I}}
end
TensorKit.scalartype(::Type{<:TransferRepresentation{T}}) where {T} = T

"""
    anyon_insertion_rep(V, op)
    anyon_insertion_rep(V)(op)(U)

Fix a center-graded bulk leg, then prepare one irrep per occupied sector of
the transfer operator's central state space. Apply the representation to `U`.
"""
function anyon_insertion_rep(V::GradedSpace{C}, op::SectorTransferOperator{I}) where {I<:Sector,C<:DrinfeldCenter{I}}
    plan = AnyonInsertionPlan(op)
    return TransferRepresentation(CentralTensorMapSpace(plan.boundary, V), plan)
end
anyon_insertion_rep(V::GradedSpace{C}) where {I<:Sector,C<:DrinfeldCenter{I}} =
    (op::SectorTransferOperator{I}) -> anyon_insertion_rep(V, op)

function TransferRepresentation(S::CentralTensorMapSpace{I,C}, plan::AnyonInsertionPlan{I}) where {I<:Sector,C<:DrinfeldCenter{I}}
    codomain(S) == plan.boundary || throw(SpaceMismatch("incompatible central state boundary"))
    irreps = Dict{C,TransferIrrep{I}}(a => TransferIrrep(a, plan) for a in blocksectors(S))
    T = promote_type(map(scalartype, values(plan.lattice.tensors))...)
    for irrep in values(irreps), h in get(irrep.coloring.tensors, :gamma, ())
        T = promote_type(T, scalartype(h))
    end
    return TransferRepresentation{T,I,C,typeof(S)}(S, irreps)
end

# Preserve the full bulk product space (including zero or several bulk legs).
function (op::SectorTransferOperator{I})(U::CentralTensorMap{T,A,S}) where {I<:Sector,T,A,S<:CentralTensorMapSpace{I}}
    return TransferRepresentation(space(U), AnyonInsertionPlan(op))(U)
end

function (rep::TransferRepresentation{T,I,C})(U::CentralTensorMap{E,A,S}) where {T,I,C,E,A,S<:CentralTensorMapSpace{I,C}}
    space(U) == rep.space || throw(SpaceMismatch("central state differs from the representation space"))
    if length(rep.irreps) == 1
        a, irrep = only(rep.irreps)
        if size(block(U,a), 2) == 1
            ψ = TensorMap{E}(U.data, irrep.space)
            return CentralTensorMap(irrep(ψ).data, rep.space)
        end
    end
    result = CentralTensorMap{promote_type(E,T)}(undef, rep.space)
    for (a, irrep) in rep.irreps
        input, output = block(U,a), block(result,a)
        for j in axes(input,2)
            ψ = TensorMap{E}(input[:,j], irrep.space)
            output[:,j] .= irrep(ψ).data
        end
    end
    return result
end
