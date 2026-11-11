"""
    lattice_geometry(graph::TensorGraph) -> LatticeGeometry

Give each base rank-(2,2) node a unit square with ports `[left up; down right]`.
Close output i onto input i, then infer the spatial and temporal periods from
this gluing. Node views carry their original side correspondence.

The time cycle crosses the output-to-input seam once. The space cycle has zero
net seam crossings. Choose a positive period determinant and reduce time modulo
space so that `-1/2 < real(tau) <= 1/2`; do not exchange the two cycles.

Tile centers follow expanded occurrence order, with the first occurrence fixing
the origin. Colors do not change the metric or enforce a checkerboard pattern.
Invalid square-torus gluings and cuts throw `ArgumentError`. No numerical tensors
are constructed. Area counts unit tiles, not the additive area term in log Z.
"""
function lattice_geometry(graph::TensorGraph)
    numout(graph) == numin(graph) && numout(graph) > 0 ||
        throw(ArgumentError("lattice geometry requires equal nonempty input/output boundaries"))

    # 1. Glue the square sides, marking crossings of the input/output seam.
    adjacency = _square_adjacency(expand(graph))
    _check_square_torus(adjacency)

    # 2. Place vertices along a spanning tree. Other edges close walks whose
    #    coordinate mismatch gives a periodic displacement, not an error.
    placement = _place_along_spanning_tree(adjacency)
    cycles = _closed_walk_displacements(adjacency, placement)

    # 3. Find one cycle crossing the seam once, and a basic cycle crossing zero times.
    time_period = _find_time_period(cycles)
    space_period = _find_space_period(cycles, time_period)

    # 4. Choose the documented basis and move tile centers into its fundamental cell.
    space_period, time_period = _normalize_period_basis(space_period, time_period)
    positions = _wrap_tile_centers(placement.positions, space_period, time_period)
    return LatticeGeometry(space_period, time_period, positions)
end

"""Glue the annular wires before inferring the unit-square periods."""
lattice_geometry(tube::TubeGraph) = lattice_geometry(closed_graph(tube))
lattice_geometry(op::TransferOperator) = lattice_geometry(op.tube)

# Neighbor columns are right, up, left, down. Base gate ports have the different
# order [left up; down right]; views permute this side correspondence explicitly.
const _square_steps = ((1, 0), (0, 1), (-1, 0), (0, -1))
_square_side_normals(::TensorNode{2,2}) = ((-1, 0), (0, 1), (0, -1), (1, 0))
_square_side_normals(::AbstractTensorNode) =
    throw(ArgumentError("unit-square geometry requires rank-(2,2) base nodes"))

function _square_side_normals(node::Union{PermutedNode,TransposedNode})
    normals = _square_side_normals(parent(node))
    return map(index -> normals[index], (node.indices[1]..., node.indices[2]...))
end

function _square_side_normals(node::AdjointNode)
    normals = _square_side_normals(parent(node))
    output_count = numout(parent(node))
    return (normals[output_count+1:end]..., normals[1:output_count]...)
end

function _square_adjacency(flat_graph)
    terms = occurrences(flat_graph)
    side_normals = map(term -> _square_side_normals(term.object), terms)
    index_ports = Dict{Int,Vector{Tuple{Int,Int}}}()
    for (vertex, term) in enumerate(terms), (port, index) in enumerate(term.indices)
        push!(get!(index_ports, index, Tuple{Int,Int}[]), (vertex, port))
    end
    neighbors = zeros(Int, length(terms), 4)
    seam_crossings = zeros(Int, length(terms), 4)
    function connect_sides!(first, second, crossings)
        source, source_port = first
        target, target_port = second
        source_normal = side_normals[source][source_port]
        target_normal = side_normals[target][target_port]
        source_normal == .-target_normal || throw(ArgumentError(
            "connected square sides must face opposite directions; cannot glue node $source port $source_port to node $target port $target_port by translation"))
        source_direction = findfirst(==(source_normal), _square_steps)
        target_direction = findfirst(==(target_normal), _square_steps)
        neighbors[source, source_direction] = target
        neighbors[target, target_direction] = source
        seam_crossings[source, source_direction] = crossings
        seam_crossings[target, target_direction] = -crossings
    end
    for ports in values(index_ports)
        length(ports) == 2 && connect_sides!(ports[1], ports[2], 0)
    end
    outputs, inputs = boundary(flat_graph)
    for i in eachindex(outputs)
        connect_sides!(only(index_ports[outputs[i]]), only(index_ports[inputs[i]]), 1)
    end
    return (; neighbors, seam_crossings)
end

function _check_square_torus(adjacency)
    neighbors, crossings = adjacency.neighbors, adjacency.seam_crossings
    for vertex in axes(neighbors, 1)
        right, up = neighbors[vertex, 1], neighbors[vertex, 2]
        # Right then up and up then right must reach the same square, with the
        # same seam count. Otherwise a unit plaquette has curvature or a cut endpoint.
        neighbors[right, 2] == neighbors[up, 1] || throw(ArgumentError(
            "horizontal and vertical translations do not commute; this is not a flat square torus"))
        crossings[vertex, 1] + crossings[right, 2] == crossings[vertex, 2] + crossings[up, 1] ||
            throw(ArgumentError("the input/output boundary does not define a closed spatial cut"))
    end
end

function _place_along_spanning_tree(adjacency)
    neighbors, crossings = adjacency.neighbors, adjacency.seam_crossings
    vertex_count = size(neighbors, 1)
    positions = fill((0, 0), vertex_count)
    crossings_from_origin = zeros(Int, vertex_count)
    visited = falses(vertex_count)
    visited[1] = true
    queue = [1]
    cursor = 1
    while cursor <= length(queue)
        source = queue[cursor]
        for direction in 1:4
            target = neighbors[source, direction]
            visited[target] && continue
            positions[target] = positions[source] .+ _square_steps[direction]
            crossings_from_origin[target] = crossings_from_origin[source] + crossings[source, direction]
            visited[target] = true
            push!(queue, target)
        end
        cursor += 1
    end
    all(visited) || throw(ArgumentError("the closed graph must describe one connected torus"))
    return (; positions, crossings_from_origin)
end

function _closed_walk_displacements(adjacency, placement)
    positions, counts = placement.positions, placement.crossings_from_origin
    # Walk from the origin to source along the tree, take this edge, then
    # return from target along the reversed tree path. Subtracting the two
    # tree placements gives the displacement and seam count of that closed walk.
    cycles = @NamedTuple{displacement::NTuple{2,Int}, seam_crossings::Int}[]
    for source in eachindex(positions), direction in 1:4
        target = adjacency.neighbors[source, direction]
        displacement = positions[source] .+ _square_steps[direction] .- positions[target]
        seam_crossings = counts[source] + adjacency.seam_crossings[source, direction] - counts[target]
        push!(cycles, (; displacement, seam_crossings))
    end
    return cycles
end

function _find_time_period(cycles)
    accumulated_crossings = 0
    accumulated_displacement = (0, 0)
    for cycle in cycles
        cycle.seam_crossings == 0 && continue
        # gcdx gives integers a,b satisfying
        #   a*accumulated_crossings + b*cycle.seam_crossings = common_crossings.
        # Apply the SAME combination to the displacements. For example, a
        # three-crossing walk minus a two-crossing walk crosses the seam once.
        common_crossings, a, b = gcdx(accumulated_crossings, cycle.seam_crossings)
        accumulated_displacement = a .* accumulated_displacement .+ b .* cycle.displacement
        accumulated_crossings = common_crossings
    end
    accumulated_crossings == 1 ||
        throw(ArgumentError("the boundary must define a primitive temporal winding"))
    return accumulated_displacement
end

function _find_space_period(cycles, time_period)
    direction = (0, 0)
    step_count_gcd = 0
    for cycle in cycles
        # Remove one time period per seam crossing. What remains has zero net
        # crossings and must be an integer multiple of the spatial period.
        displacement = cycle.displacement .- cycle.seam_crossings .* time_period
        displacement == (0, 0) && continue
        if direction == (0, 0)
            direction = displacement .÷ gcd(displacement...)
        end
        _det2(direction, displacement) == 0 || throw(ArgumentError("inconsistent spatial periods"))
        nonzero_axis = iszero(direction[1]) ? 2 : 1
        steps, remainder = divrem(displacement[nonzero_axis], direction[nonzero_axis])
        iszero(remainder) || throw(ArgumentError("nonintegral spatial period"))
        # E.g. if the observed walks span 4 and 6 steps along this direction,
        # their integer combinations generate a basic period of 2 steps.
        step_count_gcd = gcd(step_count_gcd, steps)
    end
    step_count_gcd > 0 || throw(ArgumentError("no independent spatial period was found"))
    return step_count_gcd .* direction
end

function _normalize_period_basis(space_period, time_period)
    if _det2(space_period, time_period) < 0
        space_period = .-space_period
    end
    _det2(space_period, time_period) > 0 || throw(ArgumentError("degenerate torus periods"))
    length_squared = _dot2(space_period, space_period)
    projection = _dot2(space_period, time_period)
    # Choose n = ceil(projection/length_squared - 1/2), then t <- t - n*s.
    # Integer arithmetic keeps the +1/2 tie exact: -1/2 < Re(tau) <= 1/2.
    spatial_turns = cld(2*projection - length_squared, 2*length_squared)
    return space_period, time_period .- spatial_turns .* space_period
end

function _wrap_tile_centers(positions, space_period, time_period)
    area = _det2(space_period, time_period)
    return map(positions) do position
        # Cramer's rule gives the two basis coordinates. Subtract their integer
        # parts to choose a representative in the half-open fundamental cell.
        space_turns = fld(_det2(position, time_period), area)
        time_turns = fld(_det2(space_period, position), area)
        position .- space_turns .* space_period .- time_turns .* time_period
    end
end
