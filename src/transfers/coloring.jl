"""
    checkerboard_coloring(graph, (black, white))
    checkerboard_coloring(graph, tensor)
    checkerboard_coloring(graph, cell)

Bind the A/B sites of a checkerboard. Each site is a square gate and each
internal bond joins opposite colors. For a tube this includes the annular
closure. The upper and lower boundaries stay open.
"""
function checkerboard_coloring(g::Union{TensorGraph,TubeGraph}, tensors::Tuple{AbstractTensorMap,AbstractTensorMap})
    graph = g isa TubeGraph ? closed_graph(g) : g
    flat = TensorGraphLab._suppress_wires(expand(graph))
    sites = Set{Symbol}()
    neighbors = Dict{Int,Tuple{Symbol,NTuple{2,Int}}}()
    for term in occurrences(flat)
        node = base_node(term.object)
        node isa TensorNode{2,2} || throw(ArgumentError("checkerboard sites must be square gates"))
        name = label(node)
        name in (:A, :B) || throw(ArgumentError("expected checkerboard labels A and B, found $name"))
        push!(sites, name)
        for (index, side) in zip(term.indices, _square_side_normals(term.object))
            if haskey(neighbors, index)
                other_name, other_side = neighbors[index]
                name != other_name || throw(ArgumentError("checkerboard bond $index connects two $name sites"))
                side == .-other_side || throw(ArgumentError("checkerboard bond $index must join opposite square sides"))
            else
                neighbors[index] = (name, side)
            end
        end
    end
    return coloring(g, (name => tensors[i] for (i, name) in enumerate((:A, :B)) if name in sites)...)
end

checkerboard_coloring(g::Union{TensorGraph,TubeGraph}, t::AbstractTensorMap) = checkerboard_coloring(g, (t, t))
checkerboard_coloring(g::Union{TensorGraph,TubeGraph}, c::UniformGates) = checkerboard_coloring(g, c.gate)
checkerboard_coloring(g::Union{TensorGraph,TubeGraph}, c::CheckerboardGates) = checkerboard_coloring(g, (c.black, c.white))
