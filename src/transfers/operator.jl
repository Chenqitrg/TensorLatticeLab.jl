"""
    TransferOperator(graph, coloring)

A bound transfer network with equal TensorKit domain and codomain spaces.
Store only the graph and its coloring; boundary spaces and tensor traits are
derived from them. Tensor data is borrowed, not copied.

Construction validates all bindings and ordered boundary spaces without
contracting or transforming tensor data. Equal dimensions alone are insufficient.
Only static queries are implemented; the operator is not yet callable.
"""
struct TransferOperator{G<:TensorGraph,C<:TensorGraphLab.Coloring}
    graph::G
    coloring::C
    function TransferOperator(graph::G, coloring::C) where {G<:TensorGraph,C<:TensorGraphLab.Coloring}
        W = space(graph, coloring)
        domain(W) == codomain(W) || throw(SpaceMismatch(
            "transfer domain $(domain(W)) must equal codomain $(codomain(W))"
        ))
        return new{G,C}(graph, coloring)
    end
end

TensorKit.space(op::TransferOperator) = space(op.graph, op.coloring)
TensorKit.space(op::TransferOperator, i::Int) = space(op)[i]
TensorKit.domain(op::TransferOperator) = domain(space(op))
TensorKit.codomain(op::TransferOperator) = codomain(space(op))
TensorKit.domain(op::TransferOperator, i::Int) = domain(op)[i]
TensorKit.codomain(op::TransferOperator, i::Int) = codomain(op)[i]
TensorKit.numout(op::TransferOperator) = numout(op.graph)
TensorKit.numin(op::TransferOperator) = numin(op.graph)
TensorKit.numind(op::TransferOperator) = numind(op.graph)
TensorKit.spacetype(op::TransferOperator) = spacetype(first(values(op.coloring.tensors)))
TensorKit.sectortype(op::TransferOperator) = sectortype(spacetype(op))
TensorKit.scalartype(op::TransferOperator) = promote_type(map(scalartype, values(op.coloring.tensors))...)

function Base.show(io::IO, op::TransferOperator)
    print(io, "TransferOperator(", numout(op), " ← ", numin(op),
          " legs, ", scalartype(op), ")")
end
