"""Plan with fixed lattice and crossing bindings, callable on a boundary state in `space`."""
struct TransferIrrep{I<:Sector,G<:TensorGraph,W<:(TensorMapSpace{S,N,1} where {S,N}),P<:EvaluationPlan,B<:Coloring{I}}
    graph::G
    space::W
    plan::P
    coloring::B
end

function (irrep::TransferIrrep)(ψ::AbstractTensorMap)
    space(ψ) == irrep.space || throw(SpaceMismatch("incompatible boundary state"))
    operands = map(irrep.plan.nodes) do node
        haskey(irrep.coloring.tensors, label(node)) || return ψ
        return TensorGraphLab._bind(node, irrep.coloring)
    end
    return irrep.plan(operands...)
end

# Plain spaces carry the unit charge in their own scalar field.
transfer_charge_space(::Type{S}, a::DrinfeldCenter{I}) where {I,S<:ElementarySpace} = BoundaryBulkKit.forget(a)
transfer_charge_space(::Type{S}, ::DrinfeldCenter{Trivial}) where {S<:Union{CartesianSpace,ComplexSpace}} = oneunit(S)

"""
    anyon_insertion_irrep(a, op)
    anyon_insertion_irrep(a)(op)(ψ)

Fix a center sector, then prepare its action for a transfer operator. The
resulting irrep can be applied to any boundary state in its input space.
"""
function anyon_insertion_irrep(a::DrinfeldCenter{I}, op::SectorTransferOperator{I}) where {I<:Sector}
    return TransferIrrep(a, AnyonInsertionPlan(op))
end
anyon_insertion_irrep(a::DrinfeldCenter{I}) where {I<:Sector} =
    (op::SectorTransferOperator{I}) -> anyon_insertion_irrep(a, op)

function TransferIrrep(a::DrinfeldCenter{I}, plan::AnyonInsertionPlan{I,S}) where {I<:Sector,S<:ElementarySpace}
    X = transfer_charge_space(S, a)
    W = plan.boundary ← X
    if !(S <: CartesianSpace) && isdual(X)
        throw(SpaceMismatch("charge orientation differs from the insertion graph"))
    end

    bindings = plan.lattice
    if !isempty(plan.annular)
        cache = Dict{S,AbstractTensorMap}()
        crossings = map(plan.annular) do V
            get!(cache, V) do
                S <: Union{CartesianSpace,ComplexSpace} && return BraidingTensor(X, V)
                return BoundaryBulkKit.HalfBraiding(a, V)
            end
        end
        bindings = Coloring(merge(plan.lattice.tensors, (; gamma=Tuple(crossings))))
    end
    return TransferIrrep(plan.graph, W, plan.evaluation, bindings)
end
