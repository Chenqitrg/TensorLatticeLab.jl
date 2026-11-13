"""
    eigsolve(op, levels, cfg::TransferMatrixConfig; rightstates=0)

Request `dim(levels, a)` eigenvalues in center sector `a`. Select the right
eigenvectors to retain with `:all`, a nonnegative count (the first N per
sector), or a dictionary such as `Dict(a => [1, 3], b => [2])`. An omitted
dictionary key retains no states in that sector. Indices start at one and
follow the eigenvalue order selected by `cfg.which`.
Zero or omitted multiplicities skip the sector, including the trivial sector.
An empty request returns an empty spectrum without running a Krylov solve.
"""
function KrylovKit.eigsolve(op::SectorTransferOperator{I}, levels::GradedSpace{C},
                            cfg::TransferMatrixConfig; rightstates=0) where {I<:Sector,C<:DrinfeldCenter{I}}
    options = solver_options(cfg, scalartype(op))
    requested = collect(TensorKit.blockdims(levels))
    indices = retained_levels(levels, rightstates)
    if isempty(requested)
        D = DiagonalTensorMap{float(scalartype(op))}(undef, levels)
        return TransferSpectrum(D, nothing, indices, Dict{C,KrylovKit.ConvergenceInfo}())
    end

    rng = Random.Xoshiro(cfg.seed)
    sector_results = map(requested) do (sector, count)
        result = solve_sector(op, sector, count, indices[sector], cfg, options, rng)
        return sector => result
    end
    results = Dict(sector_results)
    return pack_spectrum(op, levels, results, indices)
end

function retained_levels(levels::GradedSpace{C}, selection::Union{Integer,Symbol}) where {C<:DrinfeldCenter}
    if selection !== :all && !(selection isa Integer && !(selection isa Bool) && selection >= 0)
        throw(ArgumentError("rightstates must be :all, a nonnegative count, or a sector dictionary"))
    end
    indices = Dict{C,Vector{Int}}()
    for (sector, count) in TensorKit.blockdims(levels)
        last_level = selection === :all ? count : min(selection, count)
        indices[sector] = collect(1:Int(last_level))
    end
    return indices
end

function retained_levels(levels::GradedSpace{C}, selection::AbstractDict) where {C<:DrinfeldCenter}
    indices = Dict{C,Vector{Int}}(sector => Int[] for sector in sectors(levels))
    for (sector, selected) in selection
        sector isa C && haskey(indices, sector) ||
            throw(ArgumentError("rightstates contains an unrequested sector: $sector"))
        selected isa AbstractVector{<:Integer} ||
            throw(ArgumentError("expected a vector of level indices for sector $sector"))
        count = dim(levels, sector)
        for j in selected
            if j isa Bool || !(1 <= j <= count)
                throw(ArgumentError("right-state index $j outside the allowed levels 1:$count in sector $sector"))
            end
        end
        indices[sector] = sort!(unique!(collect(Int, selected)))
    end
    return indices
end

function krylov_dimension(dimension::Int, count::Int, requested::Union{Int,Nothing})
    if !(1 <= count <= dimension)
        throw(ArgumentError("requested $count levels in a sector of dimension $dimension"))
    end
    basis_size = min(dimension, something(requested, max(30, 2 * count + 1)))
    if basis_size < max(count, min(2, dimension))
        throw(ArgumentError("krylovdim is too small for $count levels in dimension $dimension"))
    end
    return basis_size
end

function solve_sector(op::SectorTransferOperator{I}, sector::C, count::Int, retained::Vector{Int},
                      cfg::TransferMatrixConfig, options, rng) where {I<:Sector,C<:DrinfeldCenter{I}}
    space = CentralTensorMapSpace(domain(op), Vect[C](sector => 1))
    basis_size = krylov_dimension(dim(space), count, cfg.solver.krylovdim)
    action = prepare_action(op, space)
    scalar = float(promote_type(scalartype(op), action.scalar))
    state_type = spacetype(op) === CartesianSpace ? scalar : complex(scalar)
    initial_state = CentralTensorMap(randn(rng, state_type, dim(space)), space)

    if isempty(retained) && cfg.which === :LM && isempty(cfg.solver.kwargs)
        algorithm = KrylovKit.Arnoldi(; krylovdim=basis_size, options...)
        _, _, eigenvalues, info = KrylovKit.schursolve(action, initial_state, count, :LM, algorithm)
        eigenvectors = nothing
    else
        eigenvalues, eigenvectors, info = KrylovKit.eigsolve(action, initial_state, count, cfg.which;
            krylovdim=basis_size, options..., cfg.solver.kwargs...)
    end
    if info.converged < count
        error("KrylovKit converged $(info.converged) of $count levels in sector $sector")
    end
    rightstates = isempty(retained) ? nothing : eigenvectors[retained]
    report = KrylovKit.ConvergenceInfo(count, info.residual[1:count], info.normres[1:count],
                                       info.numiter, info.numops)
    return (; eigenvalues=eigenvalues[1:count], rightstates, info=report)
end
