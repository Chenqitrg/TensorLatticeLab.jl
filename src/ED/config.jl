using TensorKit.Factorizations: TruncationSpace

"""Krylov solver settings. Additional KrylovKit keywords go in `kwargs`."""
Configurations.@option struct TransferArnoldi <: ExperimentLab.AbstractAlgorithmConfig
    krylovdim::Union{Nothing,Int} = nothing
    tol::Union{Nothing,Float64} = nothing
    maxiter::Int = 100
    kwargs::NamedTuple = (;)
end

"""Sector quotas and ordering, retained right states, and solver settings for transfer ED."""
Configurations.@option struct TransferMatrixConfig{C<:DrinfeldCenter,S<:GradedSpace{C},F} <: ExperimentLab.AbstractAlgorithmConfig
    trunc::TruncationSpace{S,F}
    rightstates::Union{Int,Symbol,Dict{C,Vector{Int}}} = 0
    solver::TransferArnoldi = TransferArnoldi()
    seed::Int = 0
end

# Normalize other integer types, ranges and dictionary implementations at the API boundary.
function TransferMatrixConfig(trunc::TruncationSpace{S}, rightstates::Integer,
                              solver::TransferArnoldi, seed::Int) where {C<:DrinfeldCenter,S<:GradedSpace{C}}
    rightstates isa Bool && throw(ArgumentError("rightstates must be a level count"))
    return TransferMatrixConfig(trunc, Int(rightstates), solver, seed)
end
function TransferMatrixConfig(trunc::TruncationSpace{S}, rightstates::AbstractDict,
                              solver::TransferArnoldi, seed::Int) where {C<:DrinfeldCenter,S<:GradedSpace{C}}
    indices = explicit_rightstates(trunc.space, rightstates)
    return TransferMatrixConfig(trunc, indices, solver, seed)
end

"""Validate the requested indices and preserve just the explicitly selected sectors."""
function explicit_rightstates(levels::GradedSpace{C}, selection::AbstractDict) where {C<:DrinfeldCenter}
    indices = Dict{C,Vector{Int}}()
    for (sector, selected) in selection
        if !(sector isa C && dim(levels, sector) > 0)
            throw(ArgumentError("rightstates contains an unrequested sector: $sector"))
        end
        indices[sector] = checked_level_indices(selected, sector, dim(levels, sector))
    end
    return indices
end

function checked_level_indices(selected::AbstractVector{<:Integer}, sector::DrinfeldCenter, count::Int)
    for index in selected
        if index isa Bool || !(1 <= index <= count)
            throw(ArgumentError("right-state index $index outside the allowed levels 1:$count in sector $sector"))
        end
    end
    return sort!(unique!(collect(Int, selected)))
end
checked_level_indices(::Any, sector::DrinfeldCenter, ::Int) =
    throw(ArgumentError("expected a vector of level indices for sector $sector"))

function eigenvalue_selector(trunc::TruncationSpace)
    by, rev = trunc.by, trunc.rev
    if by === abs && rev
        return :LM
    elseif by === real
        return rev ? :LR : :SR
    elseif by === imag
        return rev ? :LI : :SI
    end
    return KrylovKit.EigSorter(by; rev)
end

const TRANSFER_CONFIG_REGISTRY = ExperimentLab.ConfigRegistry(
    "TensorLatticeLab.TransferMatrixConfig" => TransferMatrixConfig,
    "TensorLatticeLab.TransferArnoldi" => TransferArnoldi,
)
ExperimentLab.config_registry(::Type{<:TransferMatrixConfig}) = TRANSFER_CONFIG_REGISTRY
ExperimentLab.config_registry(::Type{TransferArnoldi}) = TRANSFER_CONFIG_REGISTRY

function krylov_tolerance(settings::TransferArnoldi, scalar::Type{<:Number})
    for key in (:krylovdim, :tol, :maxiter)
        haskey(settings.kwargs, key) && throw(ArgumentError("set $key in its solver field"))
    end
    tol = something(settings.tol, sqrt(eps(real(float(scalar)))))
    if !(tol > 0 && settings.maxiter > 0)
        throw(ArgumentError("tol and maxiter must be positive"))
    end
    return tol
end
