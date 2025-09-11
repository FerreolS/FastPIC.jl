using FastPIC
using Test
using Aqua

@testset "FastPIC.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(FastPIC)
    end
    # Write your tests here.
end
