@with_kw struct BboxParams
    BBOX_DX_LOWER::Int = 2
    BBOX_DX_UPPER::Int = 2
    BBOX_DY_LOWER::Int = 21
    BBOX_DY_UPPER::Int = 18
    BBOX_WIDTH::Int = BBOX_DX_LOWER + 1 + BBOX_DX_UPPER
    BBOX_HEIGHT::Int = BBOX_DY_LOWER + 1 + BBOX_DY_UPPER
end


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
    extra_width::Int = 2 # extra width around bbox to consider neighboring lenslets when refining the model
    profile_loop::Int = 2 # number of outer loop of profile refinement
    lamp_cfwhms_init::VecOrMat{R} = vcat(2.5, zeros(profile_order))
    @assert size(lamp_cfwhms_init, 2) ≤ 2
    fit_profile_maxeval::Int = 10_000
    fit_profile_verbose::Bool = false
    refine_profile_verbose::Bool = false
    lamp_extract_restrict::Float64 = 0 # minimum relative amplitude of the profile to consider when extracting the spectrum


    # Spectral calibration parameters
    spectral_initial_order::Int = 2
    @assert spectral_initial_order ≥ 1
    spectral_final_order::Int = 3
    @assert spectral_final_order ≥ spectral_initial_order
    lasers_λs::Vector{R} = [987.72e-9, 1123.71e-9, 1309.37e-9, 1545.1e-9][1:nλ]
    laser_extract_restrict::Float64 = 0 # minimum relative amplitude of the profile to consider when extracting the spectrum
    spectral_recalibration_loop::Int = 2 # number of outer loop of spectral recalibration
    spectral_superres::Float64 = 2 # super-resolution factor when fitting the spectral model
    spectral_calibration_verbose::Bool = false


    ntasks::Int = 4 * Threads.nthreads()
end
