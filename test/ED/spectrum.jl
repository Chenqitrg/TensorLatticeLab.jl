import BoundaryBulkKit
import ExperimentLab
import KrylovKit
import Random

function edconfig(levels::GradedSpace; by::Function=abs, rev::Bool=true, kwargs...)
    return TransferMatrixConfig(; trunc=truncspace(levels; by, rev), kwargs...)
end

@testset "Transfer eigenvalues and indexed right states" begin
    V = Z3Space(0=>3, 1=>2)
    a = zeros(ComplexF64, V ← V)
    block(a, Z3Irrep(0)) .= [3im 1 0; 0 -2 2; 0 0 0.25]
    block(a, Z3Irrep(1)) .= [1+im 0; 0 -4]
    A = TensorNode(:A, a)
    @tube annular=() tube[o;i] := A[o;i]
    op = TransferOperator(tube, coloring(tube, A=>a))
    C = BoundaryBulkKit.take_center(Z3Irrep)
    trivial, charge = one(C), C(1, 0)

    empty_levels = Vect[C]()
    empty_spec = eigsolve(op, edconfig(empty_levels; rightstates=:all))
    @test only(domain(empty_spec.D)) == empty_levels
    @test isempty(blocksectors(empty_spec.D))
    @test isempty(empty_spec.info) && isempty(empty_spec.retained_indices)
    @test isnothing(empty_spec.U)
    only_charge = eigsolve(op, edconfig(Vect[C](trivial => 0, charge => 2, C(2, 0) => 0)))
    @test Set(keys(only_charge.info)) == Set([charge])
    @test block(only_charge.D, charge).diag ≈ [-4, 1+im]
    default = eigsolve(op, edconfig(Vect[C](trivial => 1)))
    @test Set(blocksectors(default.D)) == Set([trivial])
    @test block(default.D,trivial).diag ≈ [3im]
    @test isnothing(default.U)

    levels = Vect[C](trivial=>1, charge=>2)
    largest_real = edconfig(levels; by=real, rightstates=:all, seed=13,
        solver=TransferArnoldi(tol=1e-10, kwargs=(; eager=true, verbosity=0)))
    spec = eigsolve(op, largest_real)
    D, U, info = spec.D, spec.U, spec.info
    @test Set(blocksectors(D)) == Set((charge, trivial))
    @test length(block(D,trivial).diag) == size(block(U,trivial),2) == length(info[trivial].normres) == 1
    @test block(D,charge).diag ≈ [1+im, -4]
    @test info[charge].converged >= dim(levels, charge)
    @test op(U) ≈ U * D
    @test D isa DiagonalTensorMap && U isa BoundaryBulkKit.CentralTensorMap
    @test U isa TensorLatticeLab.CentralEigenvectors{Z3Irrep,C}
    twoleg = BoundaryBulkKit.CentralTensorMap{ComplexF64}(undef, V, levels ⊗ levels)
    @test !(twoleg isa TensorLatticeLab.CentralEigenvectors{Z3Irrep,C})
    @test sectortype(D) === C && domain(U) == domain(D)
    @test spec isa TensorLatticeLab.TransferSpectrum{Z3Irrep,C}
    @test default isa TensorLatticeLab.TransferSpectrum{Z3Irrep,C}
    @test_throws DimensionMismatch TensorLatticeLab.TransferSpectrum(D, nothing, spec.retained_indices, info)
    @test_throws DimensionMismatch TensorLatticeLab.TransferSpectrum(D, U,
        merge(spec.retained_indices, Dict(charge => [2])), info)
    @test_throws ArgumentError TensorLatticeLab.TransferSpectrum(D, U,
        merge(spec.retained_indices, Dict(charge => [1, 3])), info)
    @test_throws ArgumentError TensorLatticeLab.TransferSpectrum(D, U,
        merge(spec.retained_indices, Dict(charge => [1, 1])), info)
    @test_throws ArgumentError TensorLatticeLab.TransferSpectrum(D, U, Dict(charge => [1, 2]), info)
    @test_throws ArgumentError truncspace(flip(levels))
    config = edconfig(levels; seed=13, rightstates=Dict(charge => [2]))
    @test config isa ExperimentLab.AbstractAlgorithmConfig
    @test !hasproperty(config, :sectors)
    F = BoundaryBulkKit.take_center(FibonacciAnyon)
    wrong = unit(F)
    @test !applicable(anyon_insertion_irrep(wrong), op)
    @test !applicable(anyon_insertion_irrep, wrong, op)
    @test !applicable(anyon_insertion_rep(Vect[F](wrong=>1)), op)
    @test !applicable(anyon_insertion_rep, Vect[F](wrong=>1), op)
    wrong_config = edconfig(Vect[F](wrong => 1))
    method = which(eigsolve, (typeof(op), typeof(wrong_config)))
    @test method.module !== TensorLatticeLab
    result = eigsolve(op, config)
    @testset "Configuration round trips and replay" begin
        for cfg in (edconfig(empty_levels), config, largest_real)
            for mode in (:full, :compact)
                tree = ExperimentLab.configtree(cfg; mode)
                @test include_string(@__MODULE__, "ExperimentLab.@config " * repr(tree)) == cfg
                io = IOBuffer()
                ExperimentLab.writeconfig(io, cfg; include_defaults=(mode === :full))
                json = String(take!(io))
                for T in (typeof(cfg), TransferMatrixConfig)
                    @test ExperimentLab.readconfig(IOBuffer(json), T) == cfg
                end
            end
        end
        mktempdir() do path
            store = ExperimentLab.DataStore(path)
            ref = ExperimentLab.record_algorithm!(store, config)
            restored = ExperimentLab.load_algorithm(store, ref)
            @test restored == config
            replay = eigsolve(op, restored)
            @test replay.D ≈ result.D
            @test replay.U ≈ result.U
            @test replay.retained_indices == result.retained_indices
        end
    end

    vacuum = eigsolve(op, edconfig(Vect[C](trivial=>2)))
    @test Set(blocksectors(vacuum.D)) == Set([trivial])
    @test block(vacuum.D,trivial).diag ≈ [3im, -2]
    @test isempty(result.retained_indices[trivial])
    @test Set(blocksectors(result.D)) == Set((trivial, charge))
    @test block(result.D,trivial).diag ≈ [3im]
    @test block(result.D,charge).diag ≈ [-4, 1+im]
    @test result.retained_indices[charge] == [2]
    @test size(block(result.U,charge),2) == 1
    @test !applicable(rightstate, result, wrong, 1)
    @test_throws ArgumentError rightstate(result,charge,1)
    ψ = rightstate(result,charge,2)
    @test anyon_insertion_irrep(charge, op)(ψ) ≈ block(result.D,charge).diag[2] * ψ

    mixed = Vect[C](trivial => 3, charge => 2)
    prefix = eigsolve(op, edconfig(mixed; rightstates=2))
    @test prefix.retained_indices == Dict(trivial => [1, 2], charge => [1, 2])
    @test TensorLatticeLab.retained_levels(levels, 10) == Dict(trivial => [1], charge => [1, 2])
    selected = eigsolve(op, edconfig(mixed; rightstates=Dict(trivial => [1, 3], charge => [2])))
    @test selected.retained_indices == Dict(trivial => [1, 3], charge => [2])
    for (sector, j) in ((trivial, 1), (trivial, 3), (charge, 2))
        ψ = rightstate(selected, sector, j)
        @test anyon_insertion_irrep(sector, op)(ψ) ≈ block(selected.D, sector).diag[j] * ψ
    end
    for invalid in (-1, true, :none, Dict(charge => [3]), Dict(wrong => [1]), Dict(C(2, 0) => [1]))
        @test_throws ArgumentError eigsolve(op, edconfig(levels; rightstates=invalid))
    end

    full = eigsolve(op, edconfig(Vect[C](trivial=>3); rightstates=:all))
    @test block(full.D,trivial).diag ≈ [3im, -2, 0.25]
    @test op(full.U) ≈ full.U * full.D
    capped = Vect[C](charge=>3, C(2, 0)=>1)
    available = eigsolve(op, edconfig(capped; rightstates=:all))
    @test only(domain(available.D)) == Vect[C](charge=>2)
    @test available.retained_indices == Dict(charge => [1, 2])
    @test_throws ArgumentError eigsolve(op, edconfig(capped; rightstates=Dict(charge => [3])))
    @test_throws ArgumentError eigsolve(op, edconfig(levels; rightstates=Dict(charge => [0])))
    @test_throws MethodError edconfig(V)
    @test_throws ArgumentError eigsolve(op, edconfig(levels; solver=TransferArnoldi(krylovdim=1)))
    for (by, rev, expected) in ((real, false, -4), (imag, true, 1+im), (abs, false, 1+im))
        ordered = eigsolve(op, edconfig(Vect[C](charge=>1); by, rev))
        @test only(block(ordered.D, charge).diag) ≈ expected
    end

    # A marked one-site tube acts by the partial trace over a two-dimensional leg.
    for P in (ℂ^2, ℝ^2)
        gate = ones(Float64, P ⊗ P ← P ⊗ P)
        A = TensorNode(:A, gate)
        @tube annular=(s,) marked[o;i] := A[s o;i s]
        ordinary = TransferOperator(marked, coloring(marked, A=>gate))
        plain_center = BoundaryBulkKit.take_center(Trivial)
        entry = eigsolve(ordinary, edconfig(Vect[plain_center](unit(plain_center) => 1); rightstates=1))
        @test entry.D.data ≈ [4]
        @test norm(rightstate(entry,only(blocksectors(entry.D)),1)) ≈ 1
    end
end

@testset "A marked tube resolves flux at fixed boundary charge" begin
    R = Z3Space(1 => 1)
    P = Z3Space(0 => 1, 1 => 2)
    ring = id(ComplexF64, R)
    physical = zeros(ComplexF64, P ← P)
    block(physical, Z3Irrep(0)) .= 3
    block(physical, Z3Irrep(1)) .= [2 1; 0 -1]
    @tensor gate[s o; i t] := ring[s; t] * physical[o; i]
    A = TensorNode(:A, gate)
    @tube annular=(s,) tube[o; i] := A[s o; i s]
    op = TransferOperator(tube, coloring(tube, A => gate))
    C = BoundaryBulkKit.take_center(Z3Irrep)
    charge, dyon = C(1, 0), C(1, 1)
    levels = Vect[C](charge => 2, dyon => 2)
    spec = eigsolve(op, edconfig(levels; seed=23, rightstates=:all))
    D, U, info = spec.D, spec.U, spec.info
    @test block(D, charge).diag ≈ [2, -1]
    # This orientation crosses the marked line with the adjoint half-braiding.
    @test block(D, dyon).diag ≈ cispi(-2/3) .* block(D, charge).diag
    @test op(U) ≈ U * D
    @test first(info[dyon].residual) isa BoundaryBulkKit.CentralTensorMap
end

@testset "Hermitian transfer with complex right eigenvectors" begin
    P = ℂ^2
    h = TensorMap(ComplexF64[1 im; -im 1], P ← P)
    A = TensorNode(:A, h)
    @tube annular=() tube[o; i] := A[o; i]
    op = TransferOperator(tube, coloring(tube, A => h))
    C = BoundaryBulkKit.take_center(Trivial)
    cfg = edconfig(Vect[C](one(C) => 2); rightstates=2, solver=TransferArnoldi(kwargs=(; ishermitian=true)))
    spec = eigsolve(op, cfg)
    D, U = spec.D, spec.U
    @test eltype(D) <: Real
    @test eltype(U) <: Complex
    @test block(D, one(C)).diag ≈ [2, 0] atol=1e-12
    @test op(U) ≈ U * D
end

@testset "A modular-double eigenstate can span two boundary sectors" begin
    I = FibonacciAnyon
    P = Vect[I](:I => 1, :τ => 1)
    h = zeros(ComplexF64, P ← P)
    block(h, I(:I)) .= 3
    block(h, I(:τ)) .= 2im
    A = TensorNode(:A, h)
    @tube annular=() tube[o; i] := A[o; i]
    op = TransferOperator(tube, coloring(tube, A => h))
    C = BoundaryBulkKit.take_center(I)
    a = C(I(:τ), TimeReversed{I}(I(:τ)))
    X = BoundaryBulkKit.forget(a)
    @test Set(sectors(X)) == Set((I(:I), I(:τ)))
    cfg = edconfig(Vect[C](a => 2); seed=31, rightstates=Dict(a => [1, 2]))
    tree = ExperimentLab.configtree(cfg)
    @test include_string(@__MODULE__, "ExperimentLab.@config " * repr(tree)) == cfg
    io = IOBuffer()
    ExperimentLab.writeconfig(io, cfg)
    restored = ExperimentLab.readconfig(seekstart(io), TransferMatrixConfig)
    @test restored == cfg
    spec = eigsolve(op, restored)
    D, U, info = spec.D, spec.U, spec.info
    @test block(D, a).diag ≈ [3, 2im]
    @test info[a].converged == 2
    @test op(U) ≈ U * D
    for j in axes(block(U, a), 2)
        ψ = rightstate(spec, a, j)
        @test space(ψ) == (domain(op) ← X)
        @test norm(ψ) ≈ 1
        @test h * ψ ≈ block(D, a).diag[j] * ψ
    end
end
