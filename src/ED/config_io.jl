Configurations.from_dict(::Type{TransferArnoldi}, ::Configurations.OptionField{:kwargs},
                         ::Type{NamedTuple}, fields::AbstractDict) =
    (; (Symbol(key) => value for (key, value) in fields)...)

# Store the center type even when its graded space has no occupied sectors.
center_name(C::Type{<:DrinfeldCenter}) = repr(C; context=:module => Base)
center_name(::Type{BoundaryBulkKit.ModularDouble{I}}) where {I<:Union{FibonacciAnyon,IsingAnyon}} =
    "BoundaryBulkKit.ModularDouble{TensorKit.$(nameof(I))}"

function stored_center(::Type{TransferMatrixConfig}, name::AbstractString)
    for I in (Trivial, FibonacciAnyon, IsingAnyon)
        C = BoundaryBulkKit.take_center(I)
        name == center_name(C) && return C
    end
    cyclic = match(r"^BoundaryBulkKit\.ZNQuantumDouble\{([1-9][0-9]*)\}$", name)
    if isnothing(cyclic)
        throw(ArgumentError("unknown center $name; read with the concrete configuration type"))
    end
    order = parse(Int, only(cyclic.captures))
    return BoundaryBulkKit.ZNQuantumDouble{order}
end

function stored_center(::Type{<:TransferMatrixConfig{C}}, name::AbstractString) where {C}
    name == center_name(C) || throw(ArgumentError("expected $C, got $name"))
    return C
end

const TRANSFER_SORT_KEYS = (; abs, abs2, real, imag, identity)
function sort_key_name(trunc::TruncationSpace)
    for (key, f) in pairs(TRANSFER_SORT_KEYS)
        f === trunc.by && return string(key)
    end
    throw(ArgumentError("no configuration codec for sort key $(trunc.by)"))
end

# JSON uses one-based sector numbers, so keys survive decoding without parsing Julia expressions.
sector_index(a::C) where {C<:DrinfeldCenter} = TensorKit.findindex(values(C), a)

function Configurations.to_dict(::Type{<:TransferMatrixConfig}, trunc::TruncationSpace, ::Configurations.ToDictOption)
    return Dict(
        "center" => center_name(sectortype(trunc.space)),
        "quotas" => [[sector_index(a), count] for (a, count) in TensorKit.blockdims(trunc.space)],
        "by" => sort_key_name(trunc),
        "rev" => trunc.rev,
    )
end

function Configurations.from_dict(owner::Type{<:TransferMatrixConfig}, ::Configurations.OptionField{:trunc},
                                  ::Type, record::AbstractDict)
    C = stored_center(owner, record["center"])
    labels = values(C)
    levels = Vect[C](labels[i] => count for (i, count) in record["quotas"])
    by = TRANSFER_SORT_KEYS[Symbol(record["by"])]
    return truncspace(levels; by, rev=record["rev"])
end

# ExperimentLab normally turns dictionary keys into display strings. Preserve
# sector identity for this field using the same enumeration as the quotas.
function ExperimentLab._encode_config_value(::Type{<:TransferMatrixConfig{C}},
        selection::Dict{C,Vector{Int}}, ::ExperimentLab.ConfigRegistry,
        ::Configurations.ToDictOption, ::Int) where {C}
    ordered_sectors = sort!(collect(keys(selection)))
    indices = [[sector_index(a), selection[a]] for a in ordered_sectors]
    return Dict("center" => center_name(C), "indices" => indices)
end
function Configurations.from_dict(owner::Type{<:TransferMatrixConfig}, ::Configurations.OptionField{:rightstates},
                                  ::Type, record::AbstractDict)
    C = stored_center(owner, record["center"])
    labels = values(C)
    return Dict{C,Vector{Int}}(labels[i] => indices for (i, indices) in record["indices"])
end
Configurations.from_dict(::Type{<:TransferMatrixConfig}, ::Configurations.OptionField{:rightstates},
                         ::Type, value::AbstractString) = Symbol(value)
Configurations.from_dict(::Type{<:TransferMatrixConfig}, ::Configurations.OptionField{:rightstates},
                         ::Type, value::Integer) = value

# Config trees contain Julia constructor expressions. Use the same sector numbering as JSON.
function sector_entry_expression(sector::C, value::Union{Integer,Vector{Int}}) where {C<:DrinfeldCenter}
    label = "values($(center_name(C)))[$(sector_index(sector))]"
    return "$label => $(repr(value))"
end

function ExperimentLab.value_expression(trunc::TruncationSpace{<:GradedSpace{C}}) where {C<:DrinfeldCenter}
    entries = (sector_entry_expression(a, n) for (a, n) in TensorKit.blockdims(trunc.space))
    space = "TensorKit.Vect[$(center_name(C))]($(join(entries, ", ")))"
    by = sort_key_name(trunc)
    return "TensorKit.truncspace($space; by=$by, rev=$(trunc.rev))"
end

function ExperimentLab.value_expression(selection::Dict{C,Vector{Int}}) where {C<:DrinfeldCenter}
    entries = (sector_entry_expression(a, indices) for (a, indices) in selection)
    return "Dict{$(center_name(C)),Vector{Int}}($(join(entries, ", ")))"
end
