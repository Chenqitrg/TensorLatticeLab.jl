"""
    TransferOperator(tube, coloring)

A bound transfer network with equal TensorKit domain and codomain spaces.
The tube and its coloring determine the physical boundary spaces and tensor traits.
Annular legs are paired within the tube. Tensor data is borrowed.

Construction validates all bindings and ordered boundary spaces without
contracting or transforming tensor data. Equal dimensions alone are insufficient.
Use `prepare_action(op, anyon)` to obtain the callable map in a center sector.
"""
struct TransferOperator{T<:TubeGraph,C<:TensorGraphLab.Coloring}
    tube::T
    coloring::C
    function TransferOperator(tube::T, coloring::C) where {T<:TubeGraph,C<:TensorGraphLab.Coloring}
        W = space(tube, coloring)
        domain(W) == codomain(W) || throw(SpaceMismatch(
            "transfer domain $(domain(W)) must equal codomain $(codomain(W))"
        ))
        return new{T,C}(tube, coloring)
    end
end

TensorKit.space(op::TransferOperator) = space(op.tube, op.coloring)
TensorKit.space(op::TransferOperator, i::Int) = space(op)[i]
TensorKit.domain(op::TransferOperator) = domain(space(op))
TensorKit.codomain(op::TransferOperator) = codomain(space(op))
TensorKit.domain(op::TransferOperator, i::Int) = domain(op)[i]
TensorKit.codomain(op::TransferOperator, i::Int) = codomain(op)[i]
TensorKit.numout(op::TransferOperator) = numout(op.tube)
TensorKit.numin(op::TransferOperator) = numin(op.tube)
TensorKit.numind(op::TransferOperator) = numind(op.tube)
TensorKit.spacetype(op::TransferOperator) = spacetype(first(values(op.coloring.tensors)))
TensorKit.sectortype(op::TransferOperator) = sectortype(spacetype(op))
TensorKit.scalartype(op::TransferOperator) = promote_type(map(scalartype, values(op.coloring.tensors))...)

function Base.show(io::IO, op::TransferOperator)
    print(io, "TransferOperator(", numout(op), " ← ", numin(op),
          " legs, ", scalartype(op), ")")
end
