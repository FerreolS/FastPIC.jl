
@with_kw struct BboxParams
    BBOX_DX_LOWER::Int = 2
    BBOX_DX_UPPER::Int = 2
    BBOX_DY_LOWER::Int = 21
    BBOX_DY_UPPER::Int = 18
    BBOX_WIDTH::Int = BBOX_DX_LOWER + 1 + BBOX_DX_UPPER
    BBOX_HEIGHT::Int = BBOX_DY_LOWER + 1 + BBOX_DY_UPPER
end


@with_kw struct FastPICParams{R<:Real,Q}
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
    profile_order::Int = 2
    @assert profile_order ≥ 1
    extra_width::Int = 2
    profile_loop::Int = 2
    lamp_cfwhms_init::VecOrMat{R} = vcat(2.5, zeros(profile_order))


    # Spectral calibration parameters
    spectral_initial_order::Int = 2
    @assert spectral_initial_order ≥ 1
    spectral_final_order::Int = 3
    @assert spectral_final_order ≥ spectral_initial_order
    lasers_λs::Vector{R} = [987.72e-9, 1123.71e-9, 1309.37e-9, 1545.10e-9][1:nλ]
    LASERS_CX1_INIT = -0.6001811340726275
    LASERS_CX2_INIT = -0.3187688427580339
    LASERS_CY1_INIT = 89.9795748752424
    LASERS_CY2_INIT = -52.635157560302524

    lasers_fwhms_init::Vector{R} = [2.3, 2.4, 2.7, 2.9][1:nλ]
    @assert length(lasers_fwhms_init) == nλ


    λLAMP_RANGE::Q = LinRange(850e-9, 1600e-9, 10000) # coarse wavelength range of the instrument


    multi_thread::Bool = true
end
