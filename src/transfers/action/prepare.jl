"""A planar graph, its fixed color bindings, and the compiled contraction on sector states."""
struct PreparedAction{G<:TensorGraph,C<:Coloring,P<:EvaluationPlan}
    graph::G
    coloring::C
    plan::P
end

"""Complete the fixed coloring with one state. The prepared bindings are shared across calls."""
function TensorGraphLab.coloring(action::PreparedAction, ψ::AbstractTensorMap)
    terms = occurrences(action.graph)
    fixed = action.coloring
    state = only(t for t in terms if !haskey(fixed.tensors, label(t.object)))
    W = space(ψ)
    check_state_ports(state.object, W)
    for t in terms
        t === state && continue
        for (port, index) in enumerate(t.indices)
            i = findfirst(==(index), state.indices)
            isnothing(i) && continue
            V = TensorGraphLab._bindspace(t.object, fixed)[port]
            W[i] == dual(V) || throw(SpaceMismatch("incompatible space on state leg $i"))
        end
    end
    return Coloring(merge(fixed.tensors, (; label(state.object) => ψ)))
end

(action::PreparedAction)(ψ::AbstractTensorMap) = action.plan(coloring(action, ψ))

"""
    prepare_action(graph::TensorGraph, bindings; gamma, state_space, state_name=:ψ)

Prepare a graph from `action_graph` using a Coloring or a NamedTuple.
Bind the positive half-braiding as the gamma color or keyword: a TensorMap,
a tuple in annular order, a dictionary keyed by oriented space, or a callable
`gamma(V)`. A representative state in the bindings supplies its space and is
then released. The fixed tensors constrain connected state legs. A free leg
takes its space from the state passed to the prepared action.
"""
function prepare_action(graph::TensorGraph, c::Coloring; gamma=nothing,
                        state_space::Union{Nothing,TensorMapSpace}=nothing, state_name::Symbol=:ψ)
    # Keep the original index names when the action graph is already flat.
    if any(t -> t.object isa TensorGraph, occurrences(graph))
        graph = expand(graph)
    end
    graph = TensorGraphLab._suppress_wires(graph)
    flat = expand(graph)
    terms = occurrences(flat)
    states = filter(t -> label(t.object) === state_name, terms)
    length(states) == 1 && only(states).object isa TensorNode && numin(only(states).object) == 1 ||
        throw(ArgumentError("expected one basic state node named $state_name with one input"))
    state = only(states)
    crossings = Occurrence[t for t in terms if is_crossing(t)]
    fixed = fixed_coloring(terms, state, crossings, c, gamma, state_space)
    small = [last(state.indices); [t.indices[4] for t in crossings]]
    plan = TensorGraphLab.plan_evaluation(flat; method=:planar, small_indices=small)
    return PreparedAction(graph, fixed, plan)
end

function prepare_action(g::Union{TubeGraph,TensorGraph}, bindings::NamedTuple; gamma=nothing, kwargs...)
    if haskey(bindings, :gamma) && !(bindings.gamma isa TensorGraphLab.TensorBinding)
        isnothing(gamma) || throw(ArgumentError("gamma was supplied twice"))
        gamma = bindings.gamma
        bindings = Base.structdiff(bindings, NamedTuple{(:gamma,)})
    end
    return prepare_action(g, Coloring(bindings); gamma, kwargs...)
end

"""
    prepare_action(tube::TubeGraph, bindings; gamma, state_space, state_name=:ψ)
    prepare_action(tube::TubeGraph, c::Coloring, anyon)
    prepare_action(tube::TubeGraph, c::Coloring, X::ElementarySpace, gamma)
    prepare_action(op::TransferOperator; gamma, state_space)
    prepare_action(op::TransferOperator, anyon)
    prepare_action(op::TransferOperator, X::ElementarySpace, gamma)

Flatten the marked tube and reuse its lattice coloring. The sector overload
obtains the charge and positive half-braidings from BoundaryBulkKit.
"""
function prepare_action(tube::TubeGraph, c::Coloring; state_name::Symbol=:ψ,
                        state_space::Union{Nothing,TensorMapSpace}=nothing, gamma=nothing)
    graph = if haskey(c.tensors, state_name)
        action_graph(tube, TensorNode(state_name, c[state_name]))
    else
        dual_charge = !isnothing(state_space) && isdual(only(domain(state_space)))
        action_graph(tube; state_name, dual_charge)
    end
    return prepare_action(graph, c; gamma, state_space, state_name)
end

function prepare_action(tube::TubeGraph, c::Coloring, X::ElementarySpace, gamma)
    W = space(tube, c)
    typeof(X) === spacetype(W) || throw(SpaceMismatch("charge and tube space types differ"))
    return prepare_action(tube, c; gamma, state_space=domain(W) ← X)
end

function prepare_action(tube::TubeGraph, c::Coloring, anyon::Sector)
    BoundaryBulkKit.base_sector(anyon) === sectortype(c) ||
        throw(ArgumentError("anyon and tube have different underlying sectors"))
    return prepare_action(tube, c, BoundaryBulkKit.forget(anyon), V -> BoundaryBulkKit.HalfBraiding(anyon,V))
end

prepare_action(op::TransferOperator; kwargs...) = prepare_action(op.tube, op.coloring; kwargs...)
prepare_action(op::TransferOperator, anyon::Sector) = prepare_action(op.tube, op.coloring, anyon)
prepare_action(op::TransferOperator, X::ElementarySpace, gamma) = prepare_action(op.tube, op.coloring, X, gamma)

# Compatibility with the original direct-evaluation API.
function action_coloring(graph::TensorGraph, op::TransferOperator, ψ::AbstractTensorMap;
                         gamma, state_name::Symbol=:ψ)
    numin(ψ) == 1 && codomain(ψ) == domain(op) || throw(SpaceMismatch("incompatible anyon state"))
    apply = prepare_action(graph, op.coloring; gamma, state_space=space(ψ), state_name)
    return coloring(apply, ψ)
end
