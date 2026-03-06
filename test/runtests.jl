using FastPIC
using Test
using LazyArtifacts

@testset "FastPIC.jl" begin
    include("profile_unit_test.jl")
    include("integration_calibration_test.jl")
end
