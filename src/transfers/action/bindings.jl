# Index labels here are the consecutive integers of the expanded action graph.
# Store each space in the nondual formal direction so either end can identify it.
function record_ports!(ports::Dict{Int,ElementarySpace}, term::Occurrence, W::TensorMapSpace)
    for (i, index) in enumerate(term.indices)
        V = dualflags(term.object)[i] ? dual(W[i]) : W[i]
        get!(ports, index, V) == V || throw(SpaceMismatch("incompatible spaces at index $index"))
    end
end

function port_space(ports::Dict{Int,ElementarySpace}, term::Occurrence, i::Int)
    V = ports[term.indices[i]]
    return dualflags(term.object)[i] ? dual(V) : V
end

is_crossing(t::Occurrence) = t.object isa AdjointNode && parent(t.object) isa IndexedColor && label(t.object) === :gamma

# Resolve gamma(V) once per distinct oriented space. A single positive map
# can be reused, a tuple gives maps in annular order, and a dictionary uses V.
half_braiding(h::AbstractTensorMap, V::ElementarySpace) = h
half_braiding(h::AbstractDict, V::ElementarySpace) = h[V]
half_braiding(provider, V::ElementarySpace) = provider(V)

function bind_crossings!(ports::Dict{Int,ElementarySpace}, crossings::Vector{Occurrence}, gamma)
    gamma === nothing && throw(ArgumentError("supply positive half-braidings for color gamma"))
    gamma isa Tuple && length(gamma) != length(crossings) &&
        throw(ArgumentError("expected one half-braiding per annular slot"))
    cache = Dict{ElementarySpace,AbstractTensorMap}()
    maps = Vector{AbstractTensorMap}(undef, length(crossings))
    slots = sort([parent(t.object).slot for t in crossings])
    slots == eachindex(maps) || throw(ArgumentError("expected consecutive annular slots"))
    pending = copy(crossings)
    while !isempty(pending)
        # A physical strand can pass through adjacent crossings. Resolve it
        # from either known endpoint, then expose both endpoints to the rest.
        pos = findfirst(t -> haskey(ports, t.indices[2]) || haskey(ports, t.indices[3]), pending)
        isnothing(pos) && throw(ArgumentError("cannot infer an annular space from the fixed lattice or state"))
        t = popat!(pending, pos)
        V = haskey(ports,t.indices[2]) ? port_space(ports,t,2) : dual(port_space(ports,t,3))
        slot = parent(t.object).slot
        h = if gamma isa Tuple
            gamma[slot]
        else
            get!(cache, V) do
                value = half_braiding(gamma, V)
                value isa AbstractTensorMap || throw(ArgumentError("gamma must supply a TensorMap"))
                value
            end
        end
        h isa AbstractTensorMap && numout(h) == numin(h) == 2 ||
            throw(ArgumentError("gamma must supply two-output, two-input TensorMaps"))
        X = domain(h,1)
        space(h) == (V ⊗ X ← X ⊗ V) || throw(SpaceMismatch("invalid half-braiding space for $V"))
        TensorGraphLab._check_binding(parent(t.object).node, h)
        record_ports!(ports, t, adjoint(space(h)))
        maps[slot] = h
    end
    return Tuple(maps)
end

# Check the fixed lattice and crossing spaces once, leaving a single state
# binding open. A supplied state contributes only space metadata here.
function fixed_coloring(terms::Tuple, state::Occurrence, crossings::Vector{Occurrence},
                        c::Coloring, gamma, state_space::Union{Nothing,TensorMapSpace})
    state_name = label(state.object)
    fixed = Base.structdiff(c.tensors, NamedTuple{(state_name,)})
    if haskey(c.tensors, state_name)
        ψ = c[state_name]
        TensorGraphLab._check_binding(state.object, ψ)
        isnothing(state_space) || state_space == space(ψ) || throw(SpaceMismatch("incompatible state_space"))
        state_space = space(ψ)
    end
    ports = Dict{Int,ElementarySpace}()
    if !isnothing(state_space)
        check_state_ports(state.object, state_space)
        record_ports!(ports, state, state_space)
    end
    local_terms = filter(t -> label(t.object) !== state_name && !is_crossing(t), terms)
    required = Set(label(t.object) for t in local_terms)
    if !isempty(crossings) && haskey(fixed, :gamma)
        isnothing(gamma) || throw(ArgumentError("gamma was supplied twice"))
        gamma = fixed.gamma
        fixed = Base.structdiff(fixed, NamedTuple{(:gamma,)})
    end
    Set(keys(fixed)) == required || throw(ArgumentError("expected lattice colors $required"))
    for term in local_terms
        node = base_node(term.object)
        TensorGraphLab._check_binding(node, c[label(node)])
        record_ports!(ports, term, TensorGraphLab._bindspace(term.object, c))
    end
    if !isempty(crossings)
        maps = bind_crossings!(ports, crossings, gamma)
        fixed = merge(fixed, (; gamma=maps))
    end
    return Coloring(fixed)
end

function check_state_ports(node::TensorNode, W::TensorMapSpace)
    length(codomain(W)) == numout(node) && length(domain(W)) == 1 ||
        throw(SpaceMismatch("state space has the wrong number of ports"))
    if !(spacetype(W) <: CartesianSpace)
        flags = (map(isdual, Tuple(codomain(W)))..., !isdual(domain(W)[1]))
        flags == dualflags(node) || throw(SpaceMismatch("state space has incompatible directions"))
    end
end
