module Calibration

import ..FastPIC
import ..FastPIC: BboxParams, FastPICParams, @unpack_BboxParams, @unpack_FastPICParams

using Accessors,
    Atomix,
    AbstractFFTs,
    BandedMatrices,
    Bessels,
    ChunkSplitters,
    ConcreteStructs,
    DelimitedFiles,
    AstroFITS,
    InterpolationKernels,
    Interpolations,
    KernelAbstractions,
    LinearAlgebra,
    LinOps,
    NearestNeighbors,
    NonuniformFFTs,
    OhMyThreads,
    Optimisers,
    OptimPackNextGen,
    OptimPackNextGen.Powell.Newuoa,
    OptimPackNextGen.BraDi,
    Parameters,
    ProgressMeter,
    Random,
    SparseArrays,
    StaticArrays,
    StatsBase,
    StructuredArrays,
    TwoDimensional,
    TypeUtils,
    WeightedData,
    ZippedArrays

import Accessors: @reset
import WeightedData: ScaledL2Loss, get_value, loglikelihood

include("profile.jl")
include("profile_calibration.jl")
include("interpolations.jl")
include("spectral_calibration.jl")
include("transmission.jl")
include("lenslet_position.jl")

"""
    calibrate(lamp, lasers; calib_params::FastPICParams = FastPICParams(), valid_lenslets = trues(calib_params.NLENS))

Perform complete FastPIC calibration using lamp and laser reference data.

This is the main entry point for the FastPIC calibration pipeline, combining:
1. Spatial profile calibration from lamp data
2. Spectral outlier filtering
3. Wavelength calibration using laser lines
4. Template estimation and refinement

# Arguments
- `lamp`: 2D weighted array of lamp calibration data
- `lasers`: 2D weighted array of laser calibration data
- `calib_params::FastPICParams`: Configuration parameters (default: FastPICParams())
- `valid_lenslets`: Boolean mask for valid lenslets (default: all true)

# Returns
- `Tuple` containing:
  - `profiles`: Vector of fitted Profile objects for each lenslet
  - `lamp_spectra`: Vector of extracted lamp spectra
  - `coefs`: Wavelength calibration coefficients for each lenslet
  - `template`: Common lamp template spectrum
  - `transmission`: Transmission factors for each lenslet
  - `lλ`: Common wavelength grid

# Workflow
1. **Profile Calibration**: Fit spatial profile models to lamp data using iterative refinement
2. **Outlier Filtering**: Remove spectral outliers using robust statistics
3. **Spectral Calibration**: Establish wavelength solutions using laser lines and lamp template
4. **Quality Control**: Update valid lenslet mask based on calibration success

# Examples
```julia
# Standard calibration
profiles, spectra, coefs, template, trans, λ, valid = calibrate(lamp_data, laser_data)

# High-precision calibration with custom parameters
params = FastPICParams(profile_order=4, spectral_final_order=4)
profiles, spectra, coefs, template, trans, λ, valid = calibrate(
    lamp_data, laser_data; calib_params=params
)

# Subset calibration for specific lenslets
valid_subset = falses(18908)
valid_subset[1000:2000] .= true
profiles, spectra, coefs, template, trans, λ, valid = calibrate(
    lamp_data, laser_data; valid_lenslets=valid_subset
)
```

# Performance Notes
- Uses parallel processing with `calib_params.ntasks` workers
- Memory usage scales with number of valid lenslets and profile complexity
- Typical processing time: few minutes depending on parameters

# Quality Control
The function automatically updates the `valid_lenslets` mask, setting entries to `false` for:
- Lenslets with failed profile fits
- Lenslets with failed wavelength calibrations
- Lenslets identified as spectral outliers
- Lenslets outside detector boundaries
"""
function calibrate(lamp, lasers; calib_params::FastPIC.FastPICParams = FastPIC.FastPICParams(), valid_lenslets = nothing)
    grid, bboxes, lenslet_width, lenslet_θ = initialize_bboxes(lamp, lasers; calib_params = calib_params)
    profiles = initialize_profile(bboxes, grid; calib_params = calib_params)
    if valid_lenslets !== nothing
        profiles = profiles[valid_lenslets]
    end
    profiles, lamp_spectra = calibrate_profile(profiles, lamp, calib_params = calib_params)
    filter_spectra_outliers!(lamp_spectra; threshold = calib_params.outliers_threshold)
    profiles, template, transmission, lλ, _ = spectral_calibration(profiles, lasers, lamp_spectra, calib_params = calib_params)
    profiles = filter_nothing(profiles)
    transmission = estimate_transmission(profiles, lamp, lλ, template; transmission_threshold = calib_params.transmission_threshold)
    return profiles, template, transmission, lλ, lenslet_width, lenslet_θ
end

function filter_nothing(x::AbstractVector)
    T = Base.typesplit(eltype(x), Nothing)
    T === Union{} && return T[]
    mask = .!isnothing.(x)
    y = Vector{T}(undef, count(mask))
    @inbounds for (j, v) in enumerate(x[mask])
        y[j] = v::T
    end
    return y
end

end
