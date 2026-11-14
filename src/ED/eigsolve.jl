"""
    eigsolve(op, cfg::TransferMatrixConfig)

Solve each sector up to its multiplicity in `cfg.trunc.space` and its available
boundary dimension. Zero or omitted sectors are skipped. Eigenvalues follow
`cfg.trunc.by` and `cfg.trunc.rev`. Set `cfg.rightstates` to `:all`, a prefix
length, or a dictionary of one-based level indices per sector to retain.
"""
function KrylovKit.eigsolve(op::SectorTransferOperator{I}, cfg::TransferMatrixConfig{C}) where {I<:Sector,C<:DrinfeldCenter{I}}
    tolerance = krylov_tolerance(cfg.solver, scalartype(op))
    quotas = cfg.trunc.space
    boundary = domain(op)
    spaces = Dict(a => CentralTensorMapSpace(boundary, Vect[C](a => 1)) for a in sectors(quotas))
    levels = Vect[C](a => min(dim(quotas, a), dim(space)) for (a, space) in spaces)
    retained = retained_levels(levels, cfg.rightstates)

    if isempty(sectors(levels))
        D = DiagonalTensorMap{float(scalartype(op))}(undef, levels)
        return TransferSpectrum(D, nothing, retained, Dict{C,KrylovKit.ConvergenceInfo}())
    end

    plan = AnyonInsertionPlan(op)
    rng = Random.Xoshiro(cfg.seed)
    sector_results = map(sectors(levels)) do sector
        action = TransferRepresentation(spaces[sector], plan)
        count = dim(levels, sector)
        result = solve_sector(action, count, retained[sector], cfg, rng; tolerance)
        return sector => result
    end
    return pack_spectrum(op, levels, Dict(sector_results), retained)
end

function retained_levels(levels::GradedSpace{C}, selection::Symbol) where {C<:DrinfeldCenter}
    selection === :all || throw(ArgumentError("the only symbolic rightstates selection is :all"))
    return Dict{C,Vector{Int}}(a => collect(1:dim(levels, a)) for a in sectors(levels))
end

function retained_levels(levels::GradedSpace{C}, prefix::Integer) where {C<:DrinfeldCenter}
    if prefix isa Bool || prefix < 0
        throw(ArgumentError("rightstates must be a nonnegative level count"))
    end
    return Dict{C,Vector{Int}}(a => collect(1:min(prefix, dim(levels, a))) for a in sectors(levels))
end

function retained_levels(levels::GradedSpace{C}, selection::AbstractDict) where {C<:DrinfeldCenter}
    indices = explicit_rightstates(levels, selection)
    for sector in sectors(levels)
        get!(Vector{Int}, indices, sector)
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

function solve_sector(action::TransferRepresentation{T,I,C}, count::Int, retained::Vector{Int},
                      cfg::TransferMatrixConfig{C}, rng::Random.AbstractRNG;
                      tolerance::Real) where {T,I,C}
    space = action.space
    scalar = float(T)
    state_type = spacetype(codomain(space)) === CartesianSpace ? scalar : complex(scalar)
    initial_state = CentralTensorMap(randn(rng, state_type, dim(space)), space)

    solver = cfg.solver
    which = eigenvalue_selector(cfg.trunc)
    krylovdim = krylov_dimension(dim(space), count, solver.krylovdim)
    options = (; krylovdim, tol=tolerance, maxiter=solver.maxiter)
    if isempty(retained) && which === :LM && isempty(solver.kwargs)
        algorithm = KrylovKit.Arnoldi(; options...)
        _, _, eigenvalues, info = KrylovKit.schursolve(action, initial_state, count, which, algorithm)
        eigenvectors = nothing
    else
        eigenvalues, eigenvectors, info = KrylovKit.eigsolve(action, initial_state, count, which;
                                                             options..., solver.kwargs...)
    end
    if info.converged < count
        sector = only(blocksectors(space))
        error("KrylovKit converged $(info.converged) of $count levels in sector $sector")
    end
    rightstates = isempty(retained) ? nothing : eigenvectors[retained]
    report = KrylovKit.ConvergenceInfo(count, info.residual[1:count], info.normres[1:count],
                                       info.numiter, info.numops)
    return (; eigenvalues=eigenvalues[1:count], rightstates, info=report)
end
