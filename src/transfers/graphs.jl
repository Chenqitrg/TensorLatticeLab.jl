function _check_gates(A, B)
    for node in (A, B)
        node isa AbstractTensorNode{2,2} ||
            throw(ArgumentError("A and B must be nodes with two outputs and two inputs"))
    end
    return nothing
end

function _check_width(width)
    width >= 2 && iseven(width) ||
        throw(ArgumentError("width must be a positive even integer of at least two"))
    return nothing
end

"""
    row_graph(width; layers=1, shift=0, A, B)

Construct an untruncated, horizontally periodic row transfer. `width` is even;
`layers` is positive. A gate at column `j`, layer `k` has ports
`[left up; down right]` and color A when `j+k` is even, B otherwise.
Layers are numbered from the input boundary. The top of column `j` exits at
boundary position `mod1(j + shift, width)`; only the boundary is shifted.

Default colors are rank-(2,2) nodes named A and B with nondual codomain/domain
spaces. Supply extracted nodes or node views to retain other orientations.
For a repeating two-color checkerboard, `layers-shift` must be even. Other
parities still define a contraction but introduce a color mismatch at closure
unless the two colors are bound to the same tensor.
"""
function row_graph(width::Int; layers::Int=1, shift::Int=0, A=TensorNode(:A, (2, 2)), B=TensorNode(:B, (2, 2)))
    _check_width(width)
    _check_gates(A, B)
    layers >= 1 || throw(ArgumentError("layers must be positive"))
    offset = mod(shift, width)
    inputs = ntuple(j -> Symbol(:p, j), width)
    outputs = ntuple(j -> Symbol(:o, j), width)
    terms = Occurrence[]
    previous = inputs
    for k in 1:layers
        next = k == layers ? ntuple(j -> outputs[mod1(j + offset, width)], width) :
            ntuple(j -> Symbol(:v, k, :_, j), width)
        horizontal = ntuple(j -> Symbol(:h, k, :_, j), width)
        for j in 1:width
            node = iseven(j + k) ? A : B
            push!(terms, Occurrence(node,
                (horizontal[mod1(j - 1, width)], next[j]),
                (previous[j], horizontal[j])))
        end
        previous = next
    end
    return TensorGraph(Tuple(terms), outputs, inputs)
end

"""
    diagonal_graph(width; shift=0, A, B)

Construct a periodic two-layer brickwork transfer with an even number of
boundary legs. B acts on input pairs (1,2), (3,4), ...; A then acts on
(2,3), (4,5), ..., (width,1). `shift` cyclically moves the output boundary,
using the same sign convention as `row_graph`. No coarse graining is performed.
"""
function diagonal_graph(width::Int; shift::Int=0, A=TensorNode(:A, (2, 2)), B=TensorNode(:B, (2, 2)))
    _check_width(width)
    _check_gates(A, B)
    offset = mod(shift, width)
    inputs = ntuple(j -> Symbol(:p, j), width)
    outputs = ntuple(j -> Symbol(:o, j), width)
    middle = ntuple(j -> Symbol(:i, j), width)
    top = ntuple(j -> outputs[mod1(j + offset, width)], width)
    terms = Occurrence[]
    for j in 2:2:width
        k = mod1(j + 1, width)
        push!(terms, Occurrence(A, (top[j], top[k]), (middle[j], middle[k])))
    end
    for j in 1:2:width
        push!(terms, Occurrence(B, (middle[j], middle[j+1]), (inputs[j], inputs[j+1])))
    end
    return TensorGraph(Tuple(terms), outputs, inputs)
end

"""
    diagonal_2(; A, B)

Two-leg diagonal transfer, reference shape [sqrt(2), sqrt(2), 0].
Follow the diagonal graph family convention: B first, then A across the
periodic seam. Equivalent to `diagonal_graph(2; A, B)`.
"""
diagonal_2(; A=TensorNode(:A, (2, 2)), B=TensorNode(:B, (2, 2))) = diagonal_graph(2; A, B)

"""Two-leg row with one-site output shift; reference shape [1, 2, 1]."""
shifted_row_2(; A=TensorNode(:A, (2, 2)), B=TensorNode(:B, (2, 2))) = row_graph(2; shift=1, A, B)

"""Two layers on two boundary legs; reference shape [2, 2, 0]."""
double_row_2(; A=TensorNode(:A, (2, 2)), B=TensorNode(:B, (2, 2))) = row_graph(2; layers=2, A, B)

"""Four-leg brickwork transfer; reference shape [sqrt(2), 2sqrt(2), 0]."""
diagonal_4(; A=TensorNode(:A, (2, 2)), B=TensorNode(:B, (2, 2))) = diagonal_graph(4; A, B)

"""Four-leg row without an output shift; reference shape [1, 4, 0]."""
row_4(; A=TensorNode(:A, (2, 2)), B=TensorNode(:B, (2, 2))) = row_graph(4; A, B)

"""Four-leg row with one-site output shift; reference shape [1, 4, 1]."""
shifted_row_4(; A=TensorNode(:A, (2, 2)), B=TensorNode(:B, (2, 2))) = row_graph(4; shift=1, A, B)

"""Eight-leg row with one-site output shift, constructed without coarse graining."""
shifted_row_8(; A=TensorNode(:A, (2, 2)), B=TensorNode(:B, (2, 2))) = row_graph(8; shift=1, A, B)

"""
    tilted_4(; A, B)

Four-gate tilted auxiliary transfer, reference shape
[sqrt(2)/2, sqrt(2), sqrt(2)/2]. Boundary order follows TNRKit's
`_TMaction_2x2_NEtoSW`. Access as `TensorLatticeLab.tilted_4`.
"""
function tilted_4(; A=TensorNode(:A, (2, 2)), B=TensorNode(:B, (2, 2)))
    _check_gates(A, B)
    @graph g[o1 o2 o3 o4; p1 p2 p3 p4] :=
        B[o4 o1; v2 w3] * A[w3 o2; u3 p4] *
        A[o3 v2; p1 u2] * B[u2 u3; p2 p3]
    return g
end
