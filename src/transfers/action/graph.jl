_charge_indices(tube::TubeGraph) = ntuple(i -> Symbol(:x, i - 1), length(tube.annular) + 1)

"""
    action_graph(tube::TubeGraph, state::TensorNode) -> TensorGraph

Build the planar graph of Eq. (29). The state has one charge input and one
output for each physical input of the tube. Each `gamma'` joins one ordered
annular pair and represents the inverse of the positive, unitary half-braiding.
"""
function action_graph(tube::TubeGraph, state::TensorNode)
    numin(state) == 1 && numout(state) == numin(tube) ||
        throw(ArgumentError("expected a state with one charge input and one output per tube input"))
    colors = TensorGraphLab._colors(tube)
    name = label(state)
    (haskey(colors, name) || (!isempty(tube.annular) && (name === :gamma || haskey(colors, :gamma)))) &&
        throw(ArgumentError("state, gamma and lattice colors must have distinct names"))
    patch = expand(open_graph(tube))
    k = length(tube.annular)
    charge = _charge_indices(tube)
    dual_X = !last(dualflags(state))
    flags = dualflags(patch)
    terms = collect(Occurrence, occurrences(patch))
    for i in 1:k
        dual_V = flags[i]
        positive = TensorNode(:gamma, (dual_V, dual_X), (!dual_X, !dual_V))
        gamma = TensorGraphLab.IndexedColor(positive, i)
        push!(terms, Occurrence(gamma', (charge[i], patch.inputs[numin(tube)+i]),
                               (patch.outputs[i], charge[i+1])))
    end
    push!(terms, Occurrence(state, patch.inputs[1:numin(tube)], (first(charge),)))
    return TensorGraphLab._suppress_wires(TensorGraph(Tuple(terms), patch.outputs[k+1:end], (last(charge),)))
end

"""Open the tube with a symbolic state named `state_name` and the given charge orientation."""
function action_graph(tube::TubeGraph; state_name::Symbol=:ψ, dual_charge::Bool=false)
    patch = open_graph(tube)
    flags = dualflags(patch)
    outputs = ntuple(i -> !flags[numout(patch)+i], numin(tube))
    return action_graph(tube, TensorNode(state_name, outputs, (!dual_charge,)))
end
