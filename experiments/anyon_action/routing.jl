# A tensor's cyclic port order is outputs followed by reversed inputs.
# Cap the two transfer boundaries to inspect the resulting ribbon surface.
# Darts are (vertex, position in cyclic order); each edge has two darts.
function _transfer_ribbon(flat)
    rings = [Int[term.indices[1:numout(term.object)]...,
                 reverse(term.indices[numout(term.object)+1:end])...]
             for term in occurrences(flat)]
    outputs, inputs = boundary(flat)
    push!(rings, collect(inputs), reverse(collect(outputs)))
    return rings
end

function _ribbon_faces(rings)
    Dart = Tuple{Int,Int}
    edges = Dict{Int,Vector{Dart}}()
    for (v, ring) in enumerate(rings), (p, edge) in enumerate(ring)
        push!(get!(edges, edge, Dart[]), (v, p))
    end
    twin = Dict{Dart,Dart}()
    for ends in values(edges)
        length(ends) == 2 || throw(ArgumentError("a capped edge must have two ends"))
        a, b = ends
        twin[a], twin[b] = b, a
    end
    faces = Dict{Dart,Int}()
    face_count = 0
    for v in eachindex(rings), p in eachindex(rings[v])
        start = (v, p)
        haskey(faces, start) && continue
        face_count += 1
        dart = start
        while !haskey(faces, dart)
            faces[dart] = face_count
            w, q = twin[dart]
            dart = (w, mod1(q + 1, length(rings[w])))
        end
    end
    # Euler's formula only diagnoses genus after checking connectedness.
    visited = Set([1])
    queue = [1]
    for v in queue
        for p in eachindex(rings[v])
            w = twin[(v, p)][1]
            w in visited && continue
            push!(visited, w)
            push!(queue, w)
        end
    end
    length(visited) == length(rings) || throw(ArgumentError(
        "prepare_action requires a connected transfer ribbon graph"))
    length(rings) - length(edges) + face_count == 2 || throw(ArgumentError(
        "the ordered transfer boundaries do not bound a genus-zero cylinder"))
    return (; edges, twin, faces, face_count)
end

"""Find a shortest path of faces from the state charge leg to the output charge leg."""
function _charge_route(rings, ribbon)
    input_cap, output_cap = length(rings)-1, length(rings)
    start = ribbon.faces[(input_cap, 1)]
    target = ribbon.faces[(output_cap, 1)]
    # Each entry records the dart on the side from which the edge is crossed.
    adjacency = [Tuple{Int,Tuple{Int,Int}}[] for _ in 1:ribbon.face_count]
    for edge in sort!(collect(keys(ribbon.edges)))
        a, b = ribbon.edges[edge]
        fa, fb = ribbon.faces[a], ribbon.faces[b]
        push!(adjacency[fa], (fb, a))
        push!(adjacency[fb], (fa, b))
    end
    previous = zeros(Int, ribbon.face_count)
    crossed = fill((0, 0), ribbon.face_count)
    previous[start] = start
    queue = [start]
    for face in queue
        face == target && break
        for (next, dart) in adjacency[face]
            previous[next] != 0 && continue
            previous[next], crossed[next] = face, dart
            push!(queue, next)
        end
    end
    previous[target] != 0 || throw(ArgumentError("no charge route connects the boundaries"))
    route = Tuple{Int,Int}[]
    face = target
    while face != start
        push!(route, crossed[face])
        face = previous[face]
    end
    return reverse!(route)
end

# A ring position maps back to the original output-then-input tensor index.
_ring_port(node, p) = p <= numout(node) ? p : numind(node) + numout(node) + 1 - p

function _route_space(op, terms, spaces, dart)
    v, p = dart
    n = length(terms)
    v <= n && return spaces[v][_ring_port(terms[v].object, p)]
    v == n+1 && return domain(op, p)
    return dual(codomain(op, numout(op)+1-p))
end

function _route_dual(flat, dart)
    v, p = dart
    terms = occurrences(flat)
    flags = TensorGraphLab.dualflags
    v <= length(terms) && return flags(terms[v].object)[_ring_port(terms[v].object, p)]
    # Capping reverses each boundary port's formal arrow, even for real spaces.
    v == length(terms)+1 && return !flags(flat)[numout(flat)+p]
    return !flags(flat)[numout(flat)+1-p]
end
