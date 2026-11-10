"""
    coloring(graph, tensor::AbstractTensorMap)
    coloring(graph, (tensorA, tensorB))

Bind standard colors A/B without contraction. Bindings borrow the original
tensors; validation may temporarily transform node views and allocate copies.
A single tensor supplies both colors. A pair supplies A then B. Only colors
present in the graph are bound. For other color names or other unit cells, use
TensorGraphLab's explicit `coloring(graph, :name => tensor, ...)` interface.
Rank, duality and connected numerical spaces are checked by TensorGraphLab.
"""
coloring(g::TensorGraph, t::AbstractTensorMap) = coloring(g, (t, t))

function coloring(g::TensorGraph, tensors::Tuple{AbstractTensorMap,AbstractTensorMap})
    names = unique(label(base_node(term.object)) for term in occurrences(expand(g)))
    all(name -> name in (:A, :B), names) ||
        throw(ArgumentError("default coloring supports only A/B; provide explicit bindings for other colors"))
    bindings = map(names) do name
        name => tensors[name === :A ? 1 : 2]
    end
    return coloring(g, bindings...)
end

"""
    coloring(graph, cell::Union{UniformGates,CheckerboardGates})

Bind a uniform cell to both A/B colors, or bind a checkerboard's black and
white tensors to A and B respectively. No contraction is performed.
Plaquette cells require an explicit color assignment.
"""
coloring(graph::TensorGraph, cell::UniformGates) = coloring(graph, cell.gate)
coloring(graph::TensorGraph, cell::CheckerboardGates) =
    coloring(graph, (cell.black, cell.white))
