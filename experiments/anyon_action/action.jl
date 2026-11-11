"""
    PreparedAction

A planar transfer action with fixed local tensors, concrete half-braidings and
a compiled contraction. Call `action(psi)` for a new state. No full transfer
tensor is formed. Original local tensors are borrowed; transformed views and
half-braidings are prepared once. Rebuild after changing spaces or bindings.
"""
struct PreparedAction{G,T,F,W}
    graph::G
    tensors::T
    executor::F
    state_space::W
end

function (action::PreparedAction)(ψ::AbstractTensorMap)
    space(ψ) == action.state_space || throw(SpaceMismatch(
        "state space $(space(ψ)) must equal $(action.state_space)"))
    return Base.invokelatest(action.executor, action.tensors..., ψ)
end

"""
    prepare_action(op::TransferOperator, anyon)
    prepare_action(op::TransferOperator, X::ElementarySpace, halfbraiding)

Prepare the anyon-insertion action on states `domain(op) ← forget(anyon)`.
The lower-level form accepts the charge space X and a callable returning
`halfbraiding(V)::AbstractTensorMap` with space `V ⊗ X ← X ⊗ V`.
The callable is used only during preparation, once per distinct crossed space.

The graph's cyclic port orders and ordered boundaries specify a ribbon cylinder.
Find a shortest dual path between the two boundary gaps, insert concrete inverse
half-braidings, and compile a planar contraction. Connected genus-zero cylinders
with nonempty boundaries are supported, independently of graph family or rank.
Space compatibility is checked; the half-braiding axioms are the provider's
responsibility. The prepared contraction grows from the state whenever possible.
"""
function prepare_action(op::TransferOperator, anyon::Sector)
    BoundaryBulkKit.base_sector(anyon) === sectortype(op) || throw(ArgumentError(
        "anyon and transfer have different underlying categories"))
    X = BoundaryBulkKit.forget(anyon)
    X isa ElementarySpace || throw(ArgumentError("forget(anyon) must return a single space"))
    return prepare_action(op, X, V -> BoundaryBulkKit.HalfBraiding(anyon, V))
end

function prepare_action(op::TransferOperator, X::ElementarySpace, halfbraiding)
    numin(op) > 0 || throw(ArgumentError("prepare_action requires nonempty boundaries"))
    typeof(X) === spacetype(op) || throw(SpaceMismatch("charge and transfer space types differ"))
    flat, spaces, _ = TensorGraphLab._prepared_spaces(op.graph, op.coloring)
    terms = occurrences(flat)
    rings = _transfer_ribbon(flat)
    ribbon = _ribbon_faces(rings)
    route = _charge_route(rings, ribbon)
    action_graph, crossings = _insert_charge(op, X, halfbraiding, flat, spaces, rings, ribbon, route)
    # Replay local node views once, outside the Krylov loop.
    locals = map(term -> TensorGraphLab._bind(term.object, op.coloring), terms)
    tensors = (locals..., crossings...)
    tree = _action_contraction_tree(action_graph)
    executor = TensorGraphLab._compile(action_graph, :planar; tree)
    return PreparedAction(action_graph, tensors, executor, domain(op) ← X)
end

function _insert_charge(op, X, halfbraiding, flat, spaces, rings, ribbon, route)
    terms = occurrences(flat)
    nextedge = maximum(keys(ribbon.edges))
    fresh() = (nextedge += 1)
    charge = [fresh() for _ in 0:length(route)]
    crossing_terms = Occurrence[]
    crossings = AbstractTensorMap[]
    cache = Dict{typeof(X),AbstractTensorMap}()
    for (k, dart) in enumerate(route)
        other = ribbon.twin[dart]
        V = _route_space(op, terms, spaces, dart)
        inverse = get!(cache, V) do
            h = halfbraiding(V)
            h isa AbstractTensorMap || throw(ArgumentError("halfbraiding must return a TensorMap"))
            space(h) == (V ⊗ X ← X ⊗ V) || throw(SpaceMismatch(
                "half-braiding must have space V ⊗ X ← X ⊗ V"))
            # The provider supplies a unitary half-braiding. Materializing its
            # adjoint avoids repeating lazy fusion-tree transformations.
            copy(adjoint(h))
        end
        near, far = fresh(), fresh()
        rings[dart[1]][dart[2]] = near
        rings[other[1]][other[2]] = far
        flag = _route_dual(flat, dart)
        node = TensorNode(gensym(:gamma), 2, 2, (isdual(X), flag, !flag, !isdual(X)))
        # Inverse gamma: X ⊗ V ← V ⊗ X. The crossed dart determines
        # whether V is the original edge space or its dual.
        push!(crossing_terms, Occurrence(node, (charge[k], far), (near, charge[k+1])))
        push!(crossings, inverse)
    end
    rewritten = map(enumerate(terms)) do (v, term)
        m = numout(term.object)
        Occurrence(term.object, Tuple(rings[v][1:m]), Tuple(reverse(rings[v][m+1:end])))
    end
    flags = TensorGraphLab.dualflags(flat)
    state = TensorNode(gensym(:psi), numin(flat), 1,
        ((!flag for flag in flags[numout(flat)+1:end])..., !isdual(X)))
    state_term = Occurrence(state, Tuple(rings[end-1]), (first(charge),))
    graph = TensorGraph((rewritten..., crossing_terms..., state_term),
        Tuple(reverse(rings[end])), (last(charge),))
    return graph, Tuple(crossings)
end

function _action_contraction_tree(graph)
    terms = occurrences(graph)
    rings = [TensorKit.planar_unique2(Any[t.indices[1:numout(t.object)]...,
        reverse(t.indices[numout(t.object)+1:end])...]) for t in terms]
    trees = Any[i for i in eachindex(terms)]
    has_state = [i == length(terms) for i in eachindex(terms)]
    while length(trees) > 1
        best = nothing
        score = (typemax(Int), typemax(Int))
        for i in eachindex(trees), j in i+1:length(trees)
            isempty(intersect(rings[i], rings[j])) && continue
            # Delegate cyclic contraction compatibility to TensorKit.
            complements = TensorKit.possible_planar_complements(rings[i], rings[j])
            isempty(complements) && continue
            left, right = first(complements)
            candidate = (has_state[i] || has_state[j] ? 0 : 1, length(left)+length(right))
            if candidate < score
                score = candidate
                best = (i, j, vcat(left, right))
            end
        end
        isnothing(best) && throw(ArgumentError("no connected planar contraction was found"))
        i, j, ring = best
        trees[i] = (trees[i], trees[j])
        rings[i] = ring
        has_state[i] |= has_state[j]
        deleteat!(trees, j)
        deleteat!(rings, j)
        deleteat!(has_state, j)
    end
    return only(trees)
end

function Base.show(io::IO, action::PreparedAction)
    print(io, "PreparedAction(", numout(action.graph), " boundary legs, ",
        length(action.tensors), " fixed tensor occurrences)")
end
