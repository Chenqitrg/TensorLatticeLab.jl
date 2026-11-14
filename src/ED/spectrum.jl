# The eigenvector columns form one bulk leg graded by C.
const CentralEigenvectors{I,C} = CentralTensorMap{<:Number,<:Any,<:CentralTensorMapSpace{I,C,1}}

"""
Sector-resolved transfer spectrum. For sector `a`, column `k` of `block(U,a)`
belongs to eigenvalue `block(D,a).diag[retained_indices[a][k]]`.
An empty index vector means no right state was retained in that sector.
"""
struct TransferSpectrum{I<:Sector,C<:DrinfeldCenter{I},
                        D<:DiagonalTensorMap{<:Number,<:GradedSpace{C}},
                        U<:Union{Nothing,CentralEigenvectors{I,C}},
                        R<:KrylovKit.ConvergenceInfo}
    D::D
    U::U
    retained_indices::Dict{C,Vector{Int}}
    info::Dict{C,R}

    function TransferSpectrum(diagonal::D, states::U, indices::Dict{C,Vector{Int}},
                              info::Dict{C,R}) where {
            I<:Sector,C<:DrinfeldCenter{I},D<:DiagonalTensorMap{<:Number,<:GradedSpace{C}},
            U<:Union{Nothing,CentralEigenvectors{I,C}},R<:KrylovKit.ConvergenceInfo}
        levels = only(domain(diagonal))
        keys(indices) == keys(info) == Set(sectors(levels)) ||
            throw(ArgumentError("indices and convergence reports must cover the eigenvalue sectors"))

        for (a, js) in indices
            all(j -> 1 <= j <= dim(levels, a), js) && allunique(js) ||
                throw(ArgumentError("invalid retained level indices in sector $a: $js"))
        end

        if isnothing(states)
            all(isempty, values(indices)) || throw(DimensionMismatch("retained levels require right states"))
        else
            columns = only(domain(states))
            isdual(columns) == isdual(levels) || throw(SpaceMismatch("right-state and eigenvalue legs have different orientations"))
            all(a -> haskey(indices, a), sectors(columns)) ||
                throw(SpaceMismatch("right states contain a sector absent from the spectrum"))
            for (a, js) in indices
                dim(columns, a) == length(js) ||
                    throw(DimensionMismatch("expected $(length(js)) right-state columns in sector $a"))
            end
        end
        return new{I,C,D,U,R}(diagonal, states, indices, info)
    end
end

"""Return the retained right state at the original one-based sector level as a boundary TensorMap."""
function rightstate(result::TransferSpectrum{I,C}, a::C, j::Int) where {I<:Sector,C<:DrinfeldCenter{I}}
    retained = get(result.retained_indices, a, Int[])
    column = findfirst(==(j), retained)
    if isnothing(column)
        throw(ArgumentError("right state $j in $a was not retained; available levels: $retained"))
    end
    U = result.U
    boundary = codomain(U)
    charge = transfer_charge_space(spacetype(boundary), a)
    return TensorMap{eltype(U)}(block(U, a)[:, column], boundary ← charge)
end

function pack_spectrum(op::SectorTransferOperator{I}, levels::GradedSpace{C},
                       results::AbstractDict{C,<:NamedTuple}, indices::Dict{C,Vector{Int}}) where {I<:Sector,C<:DrinfeldCenter{I}}
    eigenvalue_type = promote_type((eltype(result.eigenvalues) for result in values(results))...)
    D = DiagonalTensorMap{eigenvalue_type}(undef, levels)
    for (sector, result) in results
        block(D, sector).diag .= result.eigenvalues
    end

    info = Dict(sector => result.info for (sector, result) in results)
    if all(isempty, values(indices))
        return TransferSpectrum(D, nothing, indices, info)
    end

    multiplicities = (sector => length(retained) for (sector, retained) in indices)
    state_leg = Vect[C](multiplicities)
    if isdual(levels)
        state_leg = flip(state_leg)
    end
    # A Hermitian solve can return real eigenvalues with complex right states.
    state_type = eigenvalue_type
    for result in values(results)
        isnothing(result.rightstates) && continue
        for state in result.rightstates
            state_type = promote_type(state_type, eltype(state))
        end
    end
    U = CentralTensorMap{state_type}(undef, domain(op), state_leg)
    for (sector, columns) in blocks(U)
        for (column, state) in enumerate(results[sector].rightstates)
            columns[:, column] .= view(block(state, sector), :, 1)
        end
    end
    return TransferSpectrum(D, U, indices, info)
end
