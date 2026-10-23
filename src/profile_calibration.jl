function calibrate_profile!(
        profiles,
        lamp::WeightedArray{T, 2};
        calib_params::FastPICParams = FastPICParams()
    ) where {T}


    @unpack_FastPICParams calib_params
    @unpack_BboxParams bbox_params

    NLENS = length(profiles)


    progressbar = calibration_verbose ? Progress(NLENS; desc = "Profiles estimation", showspeed = true) : nothing

    # @localize progress @localize profiles @localize lamp_spectra OhMyThreads.tforeach(eachindex(profiles, lamp_spectra); ntasks = ntasks) do i
    OhMyThreads.tmap!(profiles, profiles; ntasks = ntasks) do profile
        if !is_profile(profile)
            isnothing(progressbar) || next!(progressbar)
            return profile
        end
        if sum(view(lamp, profile.bbox).precision) == 0
            isnothing(progressbar) || next!(progressbar)
            return lenslet_invalid_data
        end

        try
            result = fit_profile(lamp, profile; maxeval = fit_profile_maxeval, verbose = fit_profile_verbose)
            isnothing(progressbar) || next!(progressbar)
            return result
        catch e
            @debug "Error on lenslet" exception = (e, catch_backtrace())
            isnothing(progressbar) || next!(progressbar)
            return lenslet_profile_fit_failed
        end
    end
    isnothing(progressbar) || finish!(progressbar)

    return profiles
end

function initialize_profile(
        bboxes,
        grid;
        calib_params::FastPICParams = FastPICParams(),
        centers = grid
    )

    @unpack_FastPICParams calib_params
    @unpack_BboxParams bbox_params

    NLENS = length(bboxes)
    profiles = Vector{Union{Profile{profile_precision, ndims(lamp_cfwhms_init)}, LensletError}}(undef, NLENS)
    fill!(profiles, lenslet_invalid_data)

    @inbounds for (i, bbox) in enumerate(bboxes)
        if bbox isa LensletError
        else
            profiles[i] = Profile(profile_precision, bbox, lamp_cfwhms_init, vcat(grid[1, i], zeros(profile_order)), Tuple(centers[:, i]))
        end
    end
    return profiles
end


"""
    fit_profile(data::WeightedArray{T,N}, profile::Profile{T2,M}; maxeval=10_000, verbose=false, relative=false) where {T,T2,N,M}

Fit a profile model to weighted data using maximum likelihood estimation.

Optimizes profile parameters by minimizing the scaled L2 loss between the profile model
and the input data. Uses the NEWUOA algorithm for derivative-free optimization.

# Arguments
- `data::WeightedArray{T,N}`: Input weighted data to fit
- `profile::Profile{T2,M}`: Initial profile model to optimize
- `maxeval::Int = 10_000`: Maximum number of function evaluations
- `verbose::Bool = false`: Enable optimization verbose output
- `relative::Bool = false`: If true, use data directly; if false, extract profile bbox

# Returns
- `Profile{T2,M}`: Optimized profile model with fitted parameters

# Algorithm
Uses automatic parameter scaling based on polynomial order and the NEWUOA 
trust-region algorithm for robust convergence.

# Examples
```julia
fitted_profile = fit_profile(lamp_data, initial_profile; maxeval=5000, verbose=true)
```
"""
function fit_profile(
        data::WeightedArray{T, N},
        profile::Profile{T2, M};
        maxeval = 10_000,
        verbose = false,
        relative = false
    ) where {T, T2, N, M}

    fwhmorder = size(profile.cfwhm, 1)
    cxorder = length(profile.cx)
    if M == 1
        scale = vcat(10.0 .^ (-(1:(fwhmorder))), 10.0 .^ (-(1:(cxorder))))
    else
        scale = vcat(10.0 .^ (-(1:(fwhmorder))), 10.0 .^ (-(1:(fwhmorder))), 10.0 .^ (-(1:(cxorder))))
    end
    vec, re = Optimisers.destructure(profile)

    d = relative ? data : view(data, profile.bbox)
    f(x) = loglikelihood(ScaledL2Loss(dims = 1, nonnegative = true), d, re(x)())
    Newuoa.optimize!(f, vec, 1.0e-2, 1.0e-9; scale = scale, check = false, maxeval = maxeval, verbose = verbose)

    any(isnan.(vec)) &&  throw("NaN found in profile for lenslet $i")

    return re(vec)
end

function estimate_shiftx(
        data::WeightedArray{T, N},
        profiles::AbstractVector{<:Union{Profile, LensletError}},
        ntasks = 4 * Threads.nthreads(),
        restrict = 0,
    ) where {T <: Real, N}


    # shiftedprofiles = deepcopy(profiles)

    loss(shift) = tmapreduce(+, profiles; outputtype = Float64, ntasks = ntasks) do profile
        tmprofile = deepcopy(profile)

        tmprofile.cx[1] += shift
        (; value, precision) = extract_spectrum(data, tmprofile; restrict = restrict, nonnegative = true)
        #return sum(precision .* value) / sum(precision .* precision)
        return sum(value .^ 2 .* precision)
    end

    shift = OptimPackNextGen.BraDi.maximize(loss, range(-0.5, 0.5; length = 10))
    # shift = Bobyqa.optimize(loss, [0.0], -1.0, 1.0, 1.0, 1.0e-9; check = false, maxeval = 100, verbose = true)
    return shift
end

function estimate_shifty(
        data::WeightedArray{T, N},
        profiles::AbstractVector{<:Union{Profile, LensletError}},
        ntasks = 4 * Threads.nthreads(),
        restrict = 0,
    ) where {T <: Real, N}


    loss(shift) = tmapreduce(+, profiles; outputtype = Float64, ntasks = ntasks) do profile
        tmprofile = deepcopy(profile)

        @reset tmprofile.ycenter = tmprofile.ycenter + shift
        (; value, precision) = extract_spectrum(data, tmprofile; restrict = restrict, nonnegative = true)
        #return sum(precision .* value) / sum(precision .* precision)
        return sum(value .^ 2 .* precision)
    end

    shift = OptimPackNextGen.BraDi.maximize(loss, range(-0.5, 0.5; length = 10))
    # shift = Bobyqa.optimize(loss, [0.0], -1.0, 1.0, 1.0, 1.0e-9; check = false, maxeval = 100, verbose = true)
    return shift
end

function estimate_shift(
        data::WeightedArray{T, N},
        profiles::AbstractVector{<:Union{Profile, LensletError}},
        ntasks = 4 * Threads.nthreads(),
        restrict = 0,
        verbose = false
    ) where {T <: Real, N}


    # shiftedprofiles = deepcopy(profiles)

    loss(shift) = tmapreduce(+, profiles; outputtype = Float64, ntasks = ntasks) do profile
        tmprofile = deepcopy(profile)
        tmprofile.cx[1] += shift[1]
        @reset tmprofile.ycenter = tmprofile.ycenter + shift[2]
        (; value, precision) = extract_spectrum(data, tmprofile; restrict = restrict, nonnegative = true)
        return sum(value .^ 2 .* precision)
    end

    #shift = OptimPackNextGen.BraDi.maximize(loss, range(-0.5, 0.5; length = 10))
    shift = Bobyqa.optimize(loss, [0.0, 0.0], -1.0, 1.0, 1.0, 1.0e-9; maximize = true, check = false, maxeval = 100, verbose = verbose)
    #shift = Newuoa.optimize!(loss, [0.0, 0.0], 1.0e-2, 1.0e-9; check = false, maxeval = 100, verbose = true)

    return shift[2]
end
