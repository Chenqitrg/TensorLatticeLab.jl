"""Krylov solver settings. Additional KrylovKit keywords go in `kwargs`."""
Configurations.@option struct TransferArnoldi <: ExperimentLab.AbstractAlgorithmConfig
    krylovdim::Union{Nothing,Int} = nothing
    tol::Union{Nothing,Float64} = nothing
    maxiter::Int = 100
    kwargs::NamedTuple = (;)
end

"""Krylov eigenvalue ordering, solver settings and random seed for transfer ED."""
Configurations.@option struct TransferMatrixConfig <: ExperimentLab.AbstractAlgorithmConfig
    which::KrylovKit.Selector = :LM
    solver::TransferArnoldi = TransferArnoldi()
    seed::Int = 0
end

const TRANSFER_CONFIG_REGISTRY = ExperimentLab.ConfigRegistry(
    "TensorLatticeLab.TransferMatrixConfig" => TransferMatrixConfig,
    "TensorLatticeLab.TransferArnoldi" => TransferArnoldi,
)
ExperimentLab.config_registry(::Type{<:TransferMatrixConfig}) = TRANSFER_CONFIG_REGISTRY
ExperimentLab.config_registry(::Type{TransferArnoldi}) = TRANSFER_CONFIG_REGISTRY

function solver_options(cfg::TransferMatrixConfig, scalar::Type)
    settings = cfg.solver
    for key in (:krylovdim, :tol, :maxiter)
        haskey(settings.kwargs, key) && throw(ArgumentError("set $key in its solver field"))
    end
    tol = something(settings.tol, sqrt(eps(real(float(scalar)))))
    if !(tol > 0 && settings.maxiter > 0)
        throw(ArgumentError("tol and maxiter must be positive"))
    end
    return (; tol, maxiter=settings.maxiter)
end
