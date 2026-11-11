# The time period on the unit-square lattice uses -1/2 < Re(tau) <= 1/2.
# Moving an output to the right moves the time period to the left.
_square_shift(shift, width) = mod(mod(shift, width) + width ÷ 2, width) - width ÷ 2

# Each internal cut is (bond, endpoint on the left of the planar strip).
# Open cuts in order from the input gap to the output gap. Where the strip
# crosses an outgoing wire, the remaining segment is a single identity.
function _transfer_tube(graph, cuts, output_shift)
    terms = collect(Occurrence, occurrences(graph))
    outputs, inputs = boundary(graph)
    endpoints = Dict(index => (v, p) for (v, t) in enumerate(terms)
                     for (p, index) in enumerate(t.indices) if index in outputs)
    wires = output_shift > 0 ? (output_shift:-1:1) : (length(outputs)+output_shift+1:length(outputs))
    for j in wires
        output = outputs[j]
        v, p = endpoints[output]
        gate = terms[v]
        link = Symbol(:through_, j)
        indices = Base.setindex(gate.indices, link, p)
        terms[v] = Occurrence(gate.object, indices[1:2], indices[3:4])
        wire = TensorGraphLab.IdentityWire(dualflags(gate.object)[p])
        push!(terms, Occurrence(wire, (output,), (link,)))
        left = output_shift > 0 ? (length(terms), 2) : (v, p)
        push!(cuts, (link, left))
    end
    left = Tuple(Symbol(:left_, a) for (a, _) in cuts)
    right = Tuple(Symbol(:right_, a) for (a, _) in cuts)
    cut_ports = Dict(a => (port, left[i], right[i]) for (i, (a, port)) in enumerate(cuts))
    patch_terms = map(enumerate(terms)) do (v, term)
        indices = ntuple(length(term.indices)) do p
            label = term.indices[p]
            haskey(cut_ports, label) || return label
            port, l, r = cut_ports[label]
            return port == (v, p) ? l : r
        end
        m = numout(term.object)
        Occurrence(term.object, indices[1:m], indices[m+1:end])
    end
    patch = TensorGraph(Tuple(patch_terms), (left..., outputs...), (inputs..., right...))
    return TubeGraph(Tuple(first.(cuts)), patch)
end
