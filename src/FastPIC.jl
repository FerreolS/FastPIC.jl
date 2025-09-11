module FastPIC

using ConcreteStructs,
    DelimitedFiles,
    EasyFITS,
    LinearAlgebra,
    OhMyThreads,
    Optimisers,
    OptimPackNextGen,
    OptimPackNextGen.Powell.Newuoa,
    Parameters,
    ProgressMeter,
    Random,
    StaticArrays,
    StatsBase,
    TwoDimensional,
    WeightedData,
    ZippedArrays


include("calibration.jl")
include("profile.jl")
include("profile_calibration.jl")
include("spectral_calibration.jl")

end
