is_crossing(t::Occurrence) = t.object isa AdjointNode && parent(t.object) isa IndexedColor && label(t.object) === :gamma

"""Sector-independent contraction data for a single colored transfer operator."""
struct AnyonInsertionPlan{I<:Sector,S<:ElementarySpace,N,
                          G<:TensorGraph{N,1},P<:EvaluationPlan,L<:Coloring{I},A<:Tuple{Vararg{S}}}
    graph::G
    evaluation::P
    lattice::L
    boundary::ProductSpace{S,N}
    annular::A
end

function AnyonInsertionPlan(op::TransferOperator)
    graph = action_graph(op.tube)
    charge_line = _charge_indices(op.tube)
    evaluation = TensorGraphLab.plan_evaluation(graph; method=:planar, small_indices=charge_line)
    annular = annular_spaces(graph, op.coloring)
    return AnyonInsertionPlan(graph, evaluation, op.coloring, domain(op), annular)
end

"""Read the space on each lattice leg in a common orientation at both ends of its edge."""
function lattice_edge_spaces(graph::TensorGraph, lattice::Coloring)
    spaces = Dict{TensorGraphLab.IndexLabel,ElementarySpace}()
    for term in occurrences(graph)
        haskey(lattice.tensors, label(term.object)) || continue
        W = TensorGraphLab._bindspace(term.object, lattice)
        for (port, edge) in enumerate(term.indices)
            V = dualflags(term.object)[port] ? dual(W[port]) : W[port]
            get!(spaces, edge, V) == V || throw(SpaceMismatch("incompatible spaces at edge $edge"))
        end
    end
    return spaces
end

"""Resolve the oriented boundary space V for each half-braiding, in seam order."""
function annular_spaces(graph::TensorGraph, lattice::Coloring)
    crossings = filter(is_crossing, occurrences(graph))
    spaces = lattice_edge_spaces(graph, lattice)
    # At γ'[x Vout; Vin y], the two V ports lie on the same strand.
    # Adjacent crossings may need a space propagated from a more distant gate.
    pending = collect(crossings)
    while !isempty(pending)
        i = findfirst(pending) do crossing
            _, out, input, _ = crossing.indices
            return haskey(spaces, out) || haskey(spaces, input)
        end
        isnothing(i) && throw(ArgumentError("cannot infer an annular space from the lattice"))
        _, out, input, _ = popat!(pending, i).indices
        V = haskey(spaces, out) ? spaces[out] : spaces[input]
        get!(spaces, out, V) == get!(spaces, input, V) || throw(SpaceMismatch("incompatible annular spaces"))
    end

    # action_graph appends one crossing per tube.annular entry in that order.
    return map(crossings) do crossing
        V = spaces[crossing.indices[2]]
        return dualflags(crossing.object)[2] ? dual(V) : V
    end
end
