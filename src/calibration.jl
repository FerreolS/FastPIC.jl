"""
    BboxParams

Configuration parameters for spectral trace bounding boxes.

Defines the spatial extent around each lenslet where spectral data is extracted
and processed. The bounding box is centered on the lenslet position with
configurable margins in both spatial dimensions.

# Fields
- `BBOX_DX_LOWER::Int = 2`: Lower margin in x-direction (pixels)
- `BBOX_DX_UPPER::Int = 2`: Upper margin in x-direction (pixels) 
- `BBOX_DY_LOWER::Int = 21`: Lower margin in y-direction (dispersion axis, pixels)
- `BBOX_DY_UPPER::Int = 18`: Upper margin in y-direction (dispersion axis, pixels)
- `BBOX_WIDTH::Int`: Total width (computed automatically)
- `BBOX_HEIGHT::Int`: Total height (computed automatically)

# Examples
```julia
# Default parameters
bbox_params = BboxParams()

```
"""
@with_kw struct BboxParams
    BBOX_DX_LOWER::Int = 2
    BBOX_DX_UPPER::Int = 2
    BBOX_DY_LOWER::Int = 21
    BBOX_DY_UPPER::Int = 18
    BBOX_WIDTH::Int = BBOX_DX_LOWER + 1 + BBOX_DX_UPPER
    BBOX_HEIGHT::Int = BBOX_DY_LOWER + 1 + BBOX_DY_UPPER
end

"""
    FastPICParams{R <: Real}

Comprehensive configuration parameters for FastPIC calibration pipeline.

Contains all settings needed for profile calibration, spectral calibration,
and data processing workflows. Parameters are organized by functionality
and include sensible defaults for typical SPHERE/IFS data.

# Type Parameters
- `R <: Real`: Numeric type for computations (typically Float32 or Float64)

# Key Parameter Groups

## Basic Configuration
- `nλ::Int = 3`: Number of laser wavelengths (3 or 4)
- `NLENS::Int = 18908`: Total number of lenslets in the detector
- `bbox_params::BboxParams`: Bounding box configuration
- `reference_pixel`: Reference pixel for wavelength calibration

## Lenslet Positions
- `LASERS_CXY0S_INIT_PATH::String`: Path to initial lenslet positions file
- `lasers_cxy0s_init::Matrix{R}`: Initial (x,y) positions for each lenslet

## Profile Calibration
- `profile_precision::Type = Float32`: Numeric precision for profile fitting
- `profile_order::Int = 3`: Polynomial order for profile parameterization
- `extra_width::Int = 2`: Extra pixels around bbox for neighboring lenslet modeling
- `profile_loop::Int = 2`: Number of profile refinement iterations
- `lamp_cfwhms_init::VecOrMat{R}`: Initial FWHM coefficients for profiles
- `fit_profile_maxeval::Int = 10_000`: Maximum evaluations for profile optimization
- `lamp_extract_restrict::Float64 = 0`: Minimum profile amplitude for extraction
- `outliers_threshold::Float64 = 3.0`: Sigma threshold for outlier filtering

## Spectral Calibration  
- `spectral_initial_order::Int = 2`: Initial polynomial order for wavelength solution
- `spectral_final_order::Int = 3`: Final polynomial order for wavelength solution
- `lasers_λs::Vector{R}`: Known laser wavelengths in meters
- `laser_extract_restrict::Float64 = 0`: Minimum amplitude for laser extraction
- `spectral_recalibration_loop::Int = 2`: Number of wavelength refinement iterations
- `spectral_superres::Float64 = 2`: Super-resolution factor for spectral modeling
- `spectral_recalibration_regul::Float64 = 1.0`: Tikhonov regularization parameter

## Performance
- `ntasks::Int`: Number of parallel tasks (default: 4 × number of threads)

# Examples
```julia
# Default parameters for standard processing
params = FastPICParams()
```

# Notes
The `@deftype R` macro ensures all numeric parameters use the specified precision type.
Assertion checks validate parameter consistency and ranges.
"""
@with_kw struct FastPICParams{R <: Real}
    @deftype R
    nλ::Int = 3
    @assert (nλ == 3 || nλ == 4)
    NLENS::Int = 18908
    @assert NLENS ≥ 1

    bbox_params::BboxParams = BboxParams()
    reference_pixel = bbox_params.BBOX_HEIGHT / 2

    # Position of the lenslets
    LASERS_CXY0S_INIT_PATH::String = joinpath(dirname(pathof(FastPIC)), "lasers_cxy0s_init.txt")
    lasers_cxy0s_init::Matrix{R} = readdlm(LASERS_CXY0S_INIT_PATH, Float64)
    @assert size(lasers_cxy0s_init) == (NLENS, 2)
    @assert all(isfinite.(lasers_cxy0s_init))

    # Profile Calibration parameters
    profile_precision::Type = Float32
    profile_order::Int = 3
    @assert profile_order ≥ 1
    extra_width::Int = 5 # extra width around bbox to consider neighboring lenslets when refining the model
    profile_loop::Int = 2 # number of outer loop of profile refinement
    lamp_cfwhms_init::VecOrMat{R} = vcat(2.5, zeros(profile_order))
    #@assert size(lamp_cfwhms_init, 2) ≤ 2
    fit_profile_maxeval::Int = 10_000
    fit_profile_verbose::Bool = false
    refine_profile_verbose::Bool = true
    lamp_extract_restrict::Float64 = 0.0 # minimum relative amplitude of the profile to consider when extracting the spectrum
    outliers_threshold::Float64 = 3.0 # threshold (in sigma) to consider a lenslet spectrum as an outlier when filtering the lamp spectra


    # Spectral calibration parameters
    spectral_initial_order::Int = 2
    @assert spectral_initial_order ≥ 1
    spectral_final_order::Int = 3
    @assert spectral_final_order ≥ spectral_initial_order
    lasers_λs::Vector{R} = [987.72e-9, 1123.71e-9, 1309.37e-9, 1545.1e-9][1:nλ]
    laser_extract_restrict::Float64 = 0.0 # minimum relative amplitude of the profile to consider when extracting the spectrum
    spectral_recalibration_loop::Int = 2 # number of outer loop of spectral recalibration
    spectral_superres::Float64 = 2 # super-resolution factor when fitting the spectral model
    spectral_calibration_verbose::Bool = true
    spectral_recalibration_regul::Float64 = 1.0 # Tikhonov regularization parameter for spectral recalibration

    ntasks::Int = 4 * Threads.nthreads()
end


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
  - `valid_lenslets`: Updated boolean mask of valid lenslets

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
function calibrate(lamp, lasers; calib_params::FastPICParams = FastPICParams(), valid_lenslets = trues(calib_params.NLENS))
    profiles, lamp_spectra = calibrate_profile(lamp, calib_params = calib_params, valid_lenslets = valid_lenslets)
    filter_spectra_outliers!(lamp_spectra; threshold = calib_params.outliers_threshold)
    coefs, template, transmission, lλ, valid_lenslets = spectral_calibration(lasers, lamp_spectra, profiles, calib_params = calib_params)

    return profiles, lamp_spectra, coefs, template, transmission, lλ, valid_lenslets
end
