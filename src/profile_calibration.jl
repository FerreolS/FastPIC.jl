"""
    calibrate_profile(lamp::WeightedArray{T,2}; calib_params::FastPICParams = FastPICParams(), valid_lenslets::AbstractVector{Bool} = trues(calib_params.NLENS)) where {T}

Calibrate spectral profiles from lamp data using iterative fitting.

This function performs a complete profile calibration workflow:
1. Initialize profile models for each valid lenslet
2. Fit individual profiles to lamp data
3. Extract initial lamp spectra
4. Refine profiles through iterative modeling to remove effects of overlapping lenslets spectra

# Arguments
- `lamp::WeightedArray{T,2}`: 2D lamp calibration data with uncertainties
- `calib_params::FastPICParams`: Configuration parameters (default: FastPICParams())
- `valid_lenslets::AbstractVector{Bool}`: Boolean mask for valid lenslets (default: all true)

# Returns
- `Tuple{Vector{Union{Profile,Nothing}}, Vector{Union{WeightedArray,Nothing}}}`: 
  - Fitted profile models for each lenslet
  - Extracted lamp spectra for each lenslet

# Examples
```julia
profiles, lamp_spectra = calibrate_profile(lamp_data; calib_params=my_params)
```
"""

function calibrate_profile(
        lamp::WeightedArray{T, 2}
        ; calib_params::FastPICParams = FastPICParams(),
        valid_lenslets::AbstractVector{Bool} = trues(calib_params.NLENS)
    ) where {T}


    @unpack_FastPICParams calib_params
    @unpack_BboxParams bbox_params

    size(valid_lenslets) == (NLENS,) || throw(ArgumentError("valid_lenslets must be of size NLENS"))


    bboxes, profiles = initialize_profile!(
        valid_lenslets,
        lamp;
        calib_params = calib_params
    )

    profile_type = ZippedVector{WeightedValue{T}, 2, true, Tuple{Vector{T}, Vector{T}}}
    lamp_spectra = Vector{Union{profile_type, Nothing}}(undef, NLENS)
    fill!(lamp_spectra, nothing)

    valid_lenslets = map(!isnothing, profiles)

    progress = Progress(sum(valid_lenslets); desc = "Profiles estimation", showspeed = true)

    @localize profiles @localize lamp_spectra OhMyThreads.tforeach(eachindex(profiles, lamp_spectra); ntasks = ntasks) do i
        if isnothing(profiles[i])
            nothing
        elseif sum(view(lamp, bboxes[i]).precision) == 0
            profiles[i] = nothing
        else
            try
                profiles[i] = fit_profile(lamp, profiles[i]; maxeval = fit_profile_maxeval, verbose = fit_profile_verbose)
                if any(isnan.(profiles[i].cfwhm))
                    throw("NaN found in profile for lenslet $i")
                end
                lamp_spectra[i] = extract_spectrum(lamp, profiles[i]; restrict = lamp_extract_restrict, nonnegative = true)

            catch e
                @debug "Error on lenslet $i" exception = (e, catch_backtrace())
                profiles[i] = nothing
            end
        end
        next!(progress)
    end
    ProgressMeter.finish!(progress)

    profiles, lamp_spectra, _ = refine_lamp_model(
        lamp,
        profiles,
        lamp_spectra;
        lamp_extract_restrict = lamp_extract_restrict,
        extra_width = extra_width,
        profile_loop = profile_loop,
        fit_profile_maxeval = fit_profile_maxeval,
        verbose = refine_profile_verbose,
        fit_profile_verbose = fit_profile_verbose,
        ntasks = ntasks
    )
    return profiles, lamp_spectra
end

"""
    initialize_profile!(valid_lenslets, lamp; calib_params::FastPICParams = FastPICParams())

Initialize profile models and bounding boxes for spectral extraction.

Creates initial profile models for each valid lenslet using configuration parameters.
Updates the `valid_lenslets` mask in-place if any lenslets are found to be invalid.

# Arguments
- `valid_lenslets`: Boolean vector indicating valid lenslets (modified in-place)
- `lamp`: Input lamp data for determining bounding boxes
- `calib_params::FastPICParams`: Configuration parameters

# Returns
- `Tuple{Vector{BoundingBox}, Vector{Union{Profile,Nothing}}}`:
  - Bounding boxes for each lenslet
  - Initial profile models for each lenslet

# Side Effects
Modifies `valid_lenslets` in-place, setting invalid lenslets to `false`.
"""
function initialize_profile!(
        valid_lenslets,
        lamp;
        calib_params::FastPICParams = FastPICParams()
    )

    @unpack_FastPICParams calib_params
    @unpack_BboxParams bbox_params


    bboxes = fill(BoundingBox{Int}(nothing), NLENS)
    profiles = Vector{Union{Profile{profile_precision, ndims(lamp_cfwhms_init)}, Nothing}}(undef, NLENS)
    fill!(profiles, nothing)

    @inbounds for i in findall(valid_lenslets)
        bbox = get_bbox(lasers_cxy0s_init[i, 1], lasers_cxy0s_init[i, 2]; bbox_params = bbox_params)
        if ismissing(bbox)
            profiles[i] = nothing
        else
            bboxes[i] = bbox
            profiles[i] = Profile(profile_precision, bbox, lamp_cfwhms_init, vcat(get_meanx(lamp, bbox), zeros(profile_order)))
        end
    end
    return bboxes, profiles
end

"""
    refine_lamp_model(lamp, profiles, lamp_spectra; kwargs...)

Refine profile models through iterative fitting and residual analysis to remove effects of overlapping spectra from neighboring lenslets.

Performs multiple iterations of profile fitting where each iteration:
1. Computes residuals from previous model
2. Refits profiles to residual + model data
3. Extracts updated spectra
4. Updates the forward model

# Arguments
- `lamp::WeightedArray{T,2}`: Input lamp calibration data
- `profiles`: Vector of initial profile models
- `lamp_spectra`: Vector of initial extracted spectra

# Keyword Arguments
- `fit_profile_maxeval::Int = 10_000`: Maximum evaluations for profile fitting
- `verbose::Bool = false`: Enable progress reporting
- `profile_loop::Int = 2`: Number of refinement iterations
- `extra_width::Int = 2`: Extra pixels around profile bbox for modeling
- `lamp_extract_restrict = 0`: Threshold for spectrum extraction
- `keep_loop::Bool = false`: Keep intermediate models for each loop
- `fit_profile_verbose::Bool = false`: Verbose output for individual fits
- `ntasks = 4*Threads.nthreads()`: Number of parallel tasks

# Returns
- `Tuple{Vector{Union{Profile,Nothing}}, Vector{Union{WeightedArray,Nothing}}, Array}`:
  - Refined profile models
  - Refined extracted spectra  
  - Forward model of the lamp data
"""
function refine_lamp_model(
        lamp,
        profiles,
        ; lamp_extract_restrict = 0,
        ntasks = 4 * Threads.nthreads(),
        kwargs...
    )
    lamp_spectra = extract_spectra(lamp, profiles; restrict = lamp_extract_restrict, ntasks = ntasks, nonnegative = true)

    return refine_lamp_model(lamp, profiles, lamp_spectra; lamp_extract_restrict = lamp_extract_restrict, ntasks = ntasks, kwargs...)
end

function refine_lamp_model(
        lamp::WeightedArray{T, 2},
        profiles,
        lamp_spectra;
        dont_fit_profile = false,
        fit_profile_maxeval = 10_000,
        verbose::Bool = false,
        profile_loop::Int = 2,
        extra_width::Int = 2,
        lamp_extract_restrict = 0,
        keep_loop::Bool = false,
        fit_profile_verbose::Bool = false,
        ntasks = 4 * Threads.nthreads()
    ) where {T}

    detectorbbox = BoundingBox(axes(lamp))

    model_value = keep_loop ? zeros(T, size(lamp)..., profile_loop) : zeros(T, size(lamp))
    model_precision = keep_loop ? fill(T(+Inf), size(lamp)..., profile_loop) : zeros(T, size(lamp))
    model_indices = LinearIndices(model_value)

    model_value_view = unsafe_wrap(AtomicMemory{T}, pointer(model_value), length(model_value); own = false)
    model_precision_view = unsafe_wrap(AtomicMemory{T}, pointer(model_precision), length(model_value); own = false)

    model = WeightedArray(model_value, model_precision)

    profiles = deepcopy(profiles)
    valid_lenslets = map(!isnothing, profiles)
    progress = nothing
    verbose && (progress = Progress(sum(valid_lenslets) .* profile_loop; desc = "Profiles refinement ($profile_loop loops)", showspeed = true))

    for l in 1:profile_loop
        if l == 1
            res = lamp
        else
            res = lamp .- (keep_loop ? view(model, :, :, l - 1) : model)
        end
        if !keep_loop
            fill!(model_value, T(0))
            fill!(model_precision, T(+Inf))
        end
        #@localize profiles @localize lamp_spectra @localize res @localize progress
        @localize res @localize progress OhMyThreads.tforeach(eachindex(profiles, lamp_spectra); ntasks = ntasks) do i
            if isnothing(profiles[i])
                nothing
            else
                try
                    if l == 1
                        resi = view(res, profiles[i].bbox)
                    else
                        prfl = (profiles[i]() .* reshape(lamp_spectra[i], 1, :))
                        resi = WeightedArray(view(res, profiles[i].bbox).value .+ prfl.value, inv.(inv.(view(res, profiles[i].bbox).precision) .- prfl.precision))
                        #    resi = WeightedArray(view(res, profiles[i].bbox).value .+ prfl.value, view(res, profiles[i].bbox).precision)
                    end
                    if !dont_fit_profile
                        profiles[i] = fit_profile(
                            deepcopy(resi),
                            deepcopy(profiles[i]);
                            relative = true,
                            maxeval = fit_profile_maxeval,
                            verbose = fit_profile_verbose
                        )
                    end
                    if any(isnan.(profiles[i].cfwhm))
                        profiles[i] = nothing
                        error("NaN found in cfwhm for lenslet $i")
                    end
                    lamp_spectra[i] = extract_spectrum(resi, profiles[i]; inbbox = true, restrict = lamp_extract_restrict, nonnegative = true)
                    if any(isnan.(lamp_spectra[i]))
                        profiles[i] = nothing
                        error("NaN found in lamp spectrum for lenslet $i")
                    end
                    (; xmin, xmax, ymin, ymax) = profiles[i].bbox
                    lbox = BoundingBox(xmin = xmin - extra_width, xmax = xmax + extra_width, ymin = ymin, ymax = ymax) ∩ detectorbbox
                    p = profiles[i](lbox)
                    if any(map(!isfinite, p))
                        profiles[i] = nothing
                        error("NaN found in profile for lenslet $i")
                    end
                    if any(map(x -> x < 0, p))
                        #profiles[i] = nothing
                        error("<0 found in profile for lenslet $i")
                    end

                    pr = p .* reshape(lamp_spectra[i], 1, :)

                    bbox_indices = keep_loop ? view(model_indices, CartesianIndices(lbox), l) : view(model_indices, CartesianIndices(lbox))
                    @inbounds for (k, idx) in enumerate(bbox_indices)
                        # e.g. Atomically accumulate into the flat `model_view`
                        if pr[k].precision > 0
                            Atomix.@atomic model_value_view[idx] += pr[k].value
                            Atomix.@atomic model_precision_view[idx] = inv(inv(pr[k].precision) + inv(model_precision_view[idx]))
                        end
                    end
                catch e
                    @debug "Error on lenslet $i" exception = e
                    #  rethrow()
                    profiles[i] = nothing
                    lamp_spectra[i] = nothing
                end
            end
            verbose && next!(progress)
        end
    end
    verbose && ProgressMeter.finish!(progress)


    model_precision[.!isfinite.(model_precision)] .= T(0)

    return profiles, lamp_spectra, model
end


"""
    get_meanx(data::WeightedArray{T,N}, bbox; relative=false) where {T,N}

Compute the precision-weighted mean position along the first axis.

Calculates the centroid position using: `Σ(x * √w * I) / Σ(√w * I)`
where x is position, w is precision (inverse variance), and I is intensity.

# Arguments
- `data::WeightedArray{T,N}`: Input weighted data
- `bbox`: Bounding box defining the region of interest
- `relative::Bool = false`: If true, use data directly; if false, extract bbox view first

# Returns
- `Float64`: Precision-weighted mean position along the first axis

# Examples
```julia
center_x = get_meanx(detector_data, profile_bbox)
```
"""
function get_meanx(data::WeightedArray{T, N}, bbox; relative = false) where {T, N}
    if relative
        (; value, precision) = data
    else
        (; value, precision) = view(data, bbox)
    end
    ax, _ = axes(bbox)

    return sum(value .* sqrt.(precision) .* ax) ./ sum(sqrt.(precision) .* value)
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
    f(x) = likelihood(ScaledL2Loss(dims = 1, nonnegative = true), d, re(x)(; normalize = false))
    Newuoa.optimize!(f, vec, 1, 1.0e-9; scale = scale, check = false, maxeval = maxeval, verbose = verbose)
    return re(vec)
end

# build_loss(data, re) = x -> likelihood(ScaledL2Loss(dims = 1, nonnegative = true), data, re(x)())

function transmission_refinement(
        spectra,
        profiles,
        transmission,
        lamp_template,
        templateλ;
        regul = 1.0,
        ntasks = 4 * Threads.nthreads()
    )

    i = 50
    nλ = length(spectra[i])

    s = build_sparse_interpolation_integration_matrix(templateλ, get_lower_uppersamples(profile_wavelength[i])...) * lamp_template
    diagA = 2 * regul * ones(Float64, nλ)
    diagA[1] = regul
    diagA[end] = regul
    diagA .+= s .^ 2 .* spectra[i].precision
    A = Array(BandedMatrix((0 => diagA, 1 => -regul * ones(nλ - 1), -1 => -regul * ones(nλ - 1)), (nλ, nλ)))
    b = s .* spectra[i].precision .* (spectra[i].value .- s)

    return A \ b
end
