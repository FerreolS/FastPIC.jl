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
    BBOX_DY_LOWER::Int = 21 #23
    BBOX_DY_UPPER::Int = 19 #16
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

    bbox_params::BboxParams = BboxParams()

    # Position of the lenslets
    template_path::String = dirname(pathof(FastPIC))
    template_file::String = nλ == 3 ? joinpath(template_path, "lamp_YJ.txt") : joinpath(template_path, "lamp_YJH.txt")
    λ_template::Vector{Float64} = readdlm(template_file, Float64)[:, 1]
    lamp_template::Vector{Float64} = readdlm(template_file, Float64)[:, 2]
    lenslets_offset::Vector{Float64} = [1024.0, 1024.0]
    lenslets_scale::Float64 = 15.5
    lenslets_θ::Float64 = 0.85
    lenslets_threshold::Float64 = 2.0

    # Profile Calibration parameters
    profile_precision::Type = Float32
    profile_order::Int = 3
    @assert profile_order ≥ 1
    extra_width::Int = 5 # extra width around bbox to consider neighboring lenslets when refining the model
    profile_loop::Int = 2 # number of outer loop of profile refinement
    lamp_cfwhms_init::VecOrMat{R} = vcat(2.35, zeros(profile_order))
    #@assert size(lamp_cfwhms_init, 2) ≤ 2
    fit_profile_maxeval::Int = 10_000
    fit_profile_verbose::Bool = false
    profile_calibration_verbose::Bool = true
    lamp_extract_restrict::Float64 = 0.0 # minimum relative amplitude of the profile to consider when extracting the spectrum
    outliers_threshold::Float64 = 3.0 # threshold (in sigma) to consider a lenslet spectrum as an outlier when filtering the lamp spectra


    # Spectral calibration parameters
    spectral_initial_order::Int = 2
    @assert spectral_initial_order ≥ 1
    spectral_final_order::Int = 3
    @assert spectral_final_order ≥ spectral_initial_order
    lasers_λs::Vector{R} = [987.72e-9, 1123.71e-9, 1309.37e-9, 1545.1e-9][1:nλ]
    laser_line_width::Vector{Float64} = fill(1.6, nλ) # FWHM of the laser lines in nm, used for fitting the laser spectra
    laser_line_pix::Vector{Float64} = nλ == 3 ? [7.0, 20.0, 35.0] : [5.0, 13, 22, 33] # initial guess for the laser line positions in pixels (relative to the center of the lenslet), used for fitting the laser spectra

    laser_extract_restrict::Float64 = 0.0 # minimum relative amplitude of the profile to consider when extracting the spectrum
    spectral_recalibration_loop::Int = 2 # number of outer loop of spectral recalibration
    spectral_superres::Float64 = 2 # super-resolution factor when fitting the spectral model
    spectral_calibration_verbose::Bool = true
    spectral_recalibration_regul::Float64 = 1.0 # Tikhonov regularization parameter for spectral recalibration

    # Position of the lenslets parameters
    position_verbose::Int = 1
    position_maxeval::Int = 1000
    position_scale::Float64 = 15.0
    position_θ::Float64 = 0.0
    position_offset::Union{Nothing, Vector{Float64}} = nothing
    position_center::Vector{Float64} = [1024.0, 1024.0]


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
function calibrate(lamp, lasers; calib_params::FastPICParams = FastPICParams(), valid_lenslets = nothing)
    grid, bboxes, lenslet_width, lenslet_θ = initialize_bboxes(lamp, lasers; calib_params = calib_params)
    profiles = initialize_profile(bboxes, grid; calib_params = calib_params)
    if valid_lenslets !== nothing
        profiles = profiles[valid_lenslets]
    end
    profiles, lamp_spectra = calibrate_profile(profiles, lamp, calib_params = calib_params)
    filter_spectra_outliers!(lamp_spectra; threshold = calib_params.outliers_threshold)
    profiles, template, transmission, lλ, _ = spectral_calibration(profiles, lasers, lamp_spectra, calib_params = calib_params)
    lenslet_index = findall(!isnothing, profiles)
    lamp_spectra = extract_spectra(lamp, profiles; restrict = 0, nonnegative = true, refinement_loop = 5)
    transmission = calibrate_spectral_transmission(lamp_spectra, profiles, template, lλ)
    profiles = filter_nothing(profiles)
    transmission = filter_nothing(transmission)
    lamp_spectra = extract_spectra(lamp, profiles; transmission = transmission, restrict = 0, nonnegative = true, refinement_loop = 5)

    return profiles, lamp_spectra, template, transmission, lλ, lenslet_index, lenslet_width
end

function filter_nothing(x::AbstractVector)
    T = Base.typesplit(eltype(x), Nothing)
    T === Union{} && return T[]

    y = Vector{T}(undef, count(!isnothing, x))
    j = 1
    @inbounds for v in x
        if v !== nothing
            y[j] = v::T
            j += 1
        end
    end
    return y
end
