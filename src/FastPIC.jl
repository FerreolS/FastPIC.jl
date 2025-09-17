module FastPIC

using BandedMatrices,
    ChunkSplitters,
    ConcreteStructs,
    DelimitedFiles,
    EasyFITS,
    InterpolationKernels,
    LinearAlgebra,
    OhMyThreads,
    Optimisers,
    OptimPackNextGen,
    OptimPackNextGen.Powell.Newuoa,
    Parameters,
    ProgressMeter,
    Random,
    SparseArrays,
    StaticArrays,
    StatsBase,
    TwoDimensional,
    WeightedData,
    ZippedArrays

export FastPICParams,
    calibrate_profile


include("calibration.jl")
include("profile.jl")
include("profile_calibration.jl")
include("interpolations.jl")
include("spectral_calibration.jl")

end
