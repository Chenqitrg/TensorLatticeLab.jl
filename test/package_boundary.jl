@testset "Tensor cells are independent of coarse-graining packages" begin
    @test parentmodule(UniformGates) === TensorLatticeLab
    @test parentmodule(CheckerboardGates) === TensorLatticeLab
    @test parentmodule(PlaquetteGates) === TensorLatticeLab
    @test parentmodule(TensorLatticeLab.assertconsistency) === TensorLatticeLab
    @test parentmodule(TensorLatticeLab.standardize) === TensorLatticeLab
    loaded = Set(pkg.name for pkg in keys(Base.loaded_modules))
    for name in ("LoopTNRLab", "PMPSLab")
        @test !(name in loaded)
    end
end
