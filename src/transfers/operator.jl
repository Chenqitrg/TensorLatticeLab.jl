"""
    TransferOperator(tube, coloring)

A bound transfer network with equal TensorKit domain and codomain spaces.
The leading type parameters `{E,S,M,N}` give the promoted scalar type,
elementary space type, and physical output and input leg counts.
TensorKit obtains the sector type as `sectortype(S)`.
Annular legs are paired within the tube. Tensor data is borrowed.

Construction validates all bindings and ordered boundary spaces without
contracting or transforming tensor data. Equal dimensions alone are insufficient.
Use `prepare_action(op, anyon)` to obtain the callable map in a center sector.
"""
struct TransferOperator{E<:Number,S<:ElementarySpace,M,N,T<:TubeGraph,C<:Coloring}
    tube::T
    coloring::C
    function TransferOperator(tube::T, coloring::C) where {T<:TubeGraph,C<:Coloring}
        W = space(tube, coloring)
        domain(W) == codomain(W) || throw(SpaceMismatch(
            "transfer domain $(domain(W)) must equal codomain $(codomain(W))"
        ))
        E = promote_type(map(scalartype, values(coloring.tensors))...)
        return new{E,spacetype(W),numout(W),numin(W),T,C}(tube, coloring)
    end
end

# Expose the coloring's sector for dispatch without duplicating it in the operator.
const SectorTransferOperator{I<:Sector} =
    TransferOperator{E,S,M,N,T,C} where {E<:Number,S<:ElementarySpace,M,N,T<:TubeGraph,C<:Coloring{I}}

TensorKit.space(op::TransferOperator) = space(op.tube, op.coloring)
TensorKit.space(op::TransferOperator, i::Int) = space(op)[i]
TensorKit.domain(op::TransferOperator) = domain(space(op))
TensorKit.codomain(op::TransferOperator) = codomain(space(op))
TensorKit.domain(op::TransferOperator, i::Int) = domain(op)[i]
TensorKit.codomain(op::TransferOperator, i::Int) = codomain(op)[i]

TensorKit.scalartype(::Type{<:TransferOperator{E}}) where {E} = E
TensorKit.spacetype(::Type{<:TransferOperator{E,S}}) where {E,S} = S
TensorKit.numout(::Type{<:TransferOperator{E,S,M}}) where {E,S,M} = M
TensorKit.numin(::Type{<:TransferOperator{E,S,M,N}}) where {E,S,M,N} = N

function Base.show(io::IO, op::TransferOperator)
    print(io, "TransferOperator(", numout(op), " ← ", numin(op),
          " legs, ", scalartype(op), ")")
end
