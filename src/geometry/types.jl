_dot2(a, b) = a[1]*b[1] + a[2]*b[2]
_det2(a, b) = a[1]*b[2] - a[2]*b[1]

"""
    LatticeGeometry(space_period, time_period, positions)

A unit-square torus with two integer period vectors and one center per tile.
`positions` follows the expanded graph's occurrence order. The periods must
have positive determinant, equal to the number of positions.

Only these three fields are stored. `area`, `circumference`, `height`, `relative_shift`
and `modular_parameter` are computed properties of the period vectors.
`relative_shift` is the exact rational displacement divided by circumference:
`dot(space_period,time_period) // dot(space_period,space_period)`.
This reference metric is distinct from geometry fitted from tensor data.
"""
struct LatticeGeometry
    space_period::NTuple{2,Int}
    time_period::NTuple{2,Int}
    positions::Vector{NTuple{2,Int}}

    function LatticeGeometry(space_period::NTuple{2,Int}, time_period::NTuple{2,Int},
                             positions::Vector{NTuple{2,Int}})
        area = _det2(space_period, time_period)
        area > 0 || throw(ArgumentError("periods must have positive oriented area"))
        area == length(positions) || throw(ArgumentError("the periods must contain one unit of area per tile"))
        return new(space_period, time_period, positions)
    end
end

function Base.getproperty(geometry::LatticeGeometry, name::Symbol)
    if name === :area
        return _det2(geometry.space_period, geometry.time_period)
    elseif name === :circumference
        return sqrt(float(_dot2(geometry.space_period, geometry.space_period)))
    elseif name === :height
        return geometry.area / geometry.circumference
    elseif name === :relative_shift
        return _dot2(geometry.space_period, geometry.time_period) //
               _dot2(geometry.space_period, geometry.space_period)
    elseif name === :modular_parameter
        length_squared = _dot2(geometry.space_period, geometry.space_period)
        projection = _dot2(geometry.space_period, geometry.time_period)
        return complex(projection / length_squared, geometry.area / length_squared)
    else
        return getfield(geometry, name)
    end
end

Base.propertynames(::LatticeGeometry, private::Bool=false) =
    (:space_period, :time_period, :positions, :area, :circumference, :height,
     :relative_shift, :modular_parameter)

function Base.show(io::IO, geometry::LatticeGeometry)
    print(io, "LatticeGeometry(space=", geometry.space_period,
          ", time=", geometry.time_period, ", tiles=", length(geometry.positions), ")")
end

function Base.show(io::IO, ::MIME"text/plain", geometry::LatticeGeometry)
    println(io, "LatticeGeometry (unit squares)")
    println(io, "  Space period:   ", geometry.space_period)
    println(io, "  Time period:    ", geometry.time_period)
    println(io, "  Area:           ", geometry.area)
    println(io, "  Circumference:  ", geometry.circumference)
    println(io, "  Height:         ", geometry.height)
    println(io, "  Relative shift: ", geometry.relative_shift)
    print(io, "  Tau:            ", geometry.modular_parameter)
end
