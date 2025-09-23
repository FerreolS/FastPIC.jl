"""
    FastPIC

Fast PIC (FastPIC).

A Julia package for calibrating SPHERE/IFS data. FastPIC provides efficient algorithms for spatial profile
modeling, spectral extraction, and wavelength calibration using lamp and laser
reference data.
It is a variant of the original PIC (Projection, Integration Convolution) from Anthony Berdeu et al. A&A 635, A90 (2020)

# Key Features

## Spatial Profile Calibration
- Parametric Gaussian-like profile models with polynomial variation
- Iterative fitting with neighboring lenslet interference modeling
- Robust outlier detection and filtering
- Support for asymmetric profiles

## Spectral Calibration  
- Laser line fitting for initial wavelength solutions
- Template-based lamp spectrum modeling
- Iterative wavelength solution refinement
- Polynomial wavelength calibration with configurable order

## Performance
- Multi-threaded parallel processing
- Memory-efficient sparse matrix operations
- Configurable precision (Float32/Float64)
- Zero-allocation inner loops for critical paths

## Data Structures
- Integration with `WeightedData.jl` for uncertainty propagation
- Support for missing/invalid lenslets
- Flexible bounding box configuration
- Quality control and validation masks

# Main Functions

## Calibration Pipeline
- [`calibrate`](@ref): Complete calibration pipeline (lamp + laser data)
- [`calibrate_profile`](@ref): Spatial profile calibration from lamp data
- [`spectral_calibration`](@ref): Wavelength calibration from laser + template data

## Utility Functions
- [`get_wavelength`](@ref): Convert pixel coordinates to wavelength
- [`FastPICParams`](@ref): Configuration parameter structure

# Typical Workflow

```julia
using FastPIC, WeightedData

# Load calibration data (lamp and laser frames)
lamp_data = WeightedArray(lamp_values, lamp_weights)
laser_data = WeightedArray(laser_values, laser_weights)

# Configure parameters
params = FastPICParams(
    profile_order = 3,          # Polynomial order for profile variation
    spectral_final_order = 4,   # Final wavelength polynomial order
    ntasks = 8                  # Parallel tasks
)

# Run complete calibration
profiles, lamp_spectra, coefs, template, transmission, λ_grid, valid = calibrate(
    lamp_data, laser_data; calib_params = params
)

# Convert pixel coordinates to wavelength for a specific lenslet
lenslet_id = 1000
pixel_coords = 1:40
wavelengths = get_wavelength(coefs[lenslet_id], params.reference_pixel, pixel_coords)

# Extract spectrum from science data using calibrated profiles
science_spectrum = extract_spectrum(science_data, profiles[lenslet_id])
```

# Algorithm References

The implementation is based on optimal extraction techniques and robust statistical
methods commonly used in astronomical spectroscopy:

- Horne, K. (1986): Optimal extraction of CCD spectra

# Dependencies

Core dependencies include:
- `WeightedData.jl`: Uncertainty propagation and weighted statistics
- `TwoDimensional.jl`: 2D geometric operations and bounding boxes
- `OptimPackNextGen.jl`: Derivative-free optimization (NEWUOA algorithm)
- `InterpolationKernels.jl`: High-quality interpolation for spectral resampling

See Project.toml for complete dependency list and version requirements.

"""
module FastPIC

using Atomix,
    BandedMatrices,
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
    calibrate_profile,
    spectral_calibration,
    calibrate,
    get_wavelength


include("calibration.jl")
include("profile.jl")
include("profile_calibration.jl")
include("interpolations.jl")
include("spectral_calibration.jl")

end
