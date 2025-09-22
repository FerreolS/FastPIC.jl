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
    #Threads.@threads for i in findall(valid_lenslets)
    # from https://discourse.julialang.org/t/optionally-multi-threaded-for-loop/81902/8?u=skleinbo
    # _foreach = multi_thread ? OhMyThreads.tforeach : Base.foreach
    @localize profiles @localize lamp_spectra OhMyThreads.tforeach(eachindex(profiles, lamp_spectra); ntasks = Threads.nthreads() * 4) do i
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
                @debug "Error on lenslet $i" exception = e
                profiles[i] = nothing
            end
        end
        next!(progress)
    end
    ProgressMeter.finish!(progress)

    profiles, lamp_spectra, model = refine_lamp_model(
        lamp,
        profiles,
        lamp_spectra;
        lamp_extract_restrict = lamp_extract_restrict,
        extra_width = extra_width,
        profile_loop = profile_loop,
        fit_profile_maxeval = fit_profile_maxeval,
        verbose = refine_profile_verbose,
        fit_profile_verbose = fit_profile_verbose
    )
    return profiles, bboxes, lamp_spectra, model
end

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
            valid_lenslets[i] = false
            profiles[i] = nothing
        else
            bboxes[i] = bbox
            profiles[i] = Profile(profile_precision, bbox, lamp_cfwhms_init, vcat(get_meanx(lamp, bbox), zeros(profile_order)))
        end
    end
    return bboxes, profiles
end

function refine_lamp_model(
        lamp,
        profiles,
        ; lamp_extract_restrict = 0,
        kwargs...
    )
    lamp_spectra = extract_spectra(lamp, profiles; restrict = lamp_extract_restrict)

    return refine_lamp_model(lamp, profiles, lamp_spectra; lamp_extract_restrict = lamp_extract_restrict, kwargs...)
end

function refine_lamp_model(
        lamp::WeightedArray{T, 2},
        profiles,
        lamp_spectra
        ; fit_profile_maxeval = 10_000,
        verbose::Bool = false,
        profile_loop::Int = 2,
        extra_width::Int = 2,
        lamp_extract_restrict = 0,
        keep_loop::Bool = false,
        fit_profile_verbose::Bool = false,
        ntasks = Threads.nthreads()
    ) where {T}

    detectorbbox = BoundingBox(axes(lamp))

    model = keep_loop ? zeros(T, size(lamp)..., profile_loop) : zeros(T, size(lamp))
    model_indices = LinearIndices(model)

    model_view = unsafe_wrap(AtomicMemory{T}, pointer(model), length(model); own = false)

    valid_lenslets = map(!isnothing, profiles)
    progress = nothing
    verbose && (progress = Progress(sum(valid_lenslets) .* profile_loop; desc = "Profiles refinement ($profile_loop loops)", showspeed = true))

    for l in 1:profile_loop
        if l == 1
            res = lamp
        else
            res = lamp .- (keep_loop ? view(model, :, :, l - 1) : model)
        end
        keep_loop || fill!(model, 0.0)
        @localize profiles @localize lamp_spectra @localize res @localize progress  OhMyThreads.tforeach(eachindex(profiles, lamp_spectra); ntasks = 1) do i
            if isnothing(profiles[i])
                nothing
            else
                try
                    if l == 1
                        resi = view(res, profiles[i].bbox)
                    else
                        resi = WeightedArray(view(res, profiles[i].bbox).value .+ profiles[i]() .* reshape(lamp_spectra[i].value, 1, :), view(res, profiles[i].bbox).precision)
                    end
                    profiles[i] = fit_profile(
                        resi, profiles[i]; relative = true, maxeval = fit_profile_maxeval, verbose = fit_profile_verbose
                    )
                    if any(isnan.(profiles[i].cfwhm))
                        profiles[i] = nothing
                        error("NaN found in cfwhm for lenslet $i")
                    end
                    lamp_spectra[i] = extract_spectrum(resi, profiles[i]; inbbox = true, restrict = lamp_extract_restrict)
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

                    pr = p .* reshape(lamp_spectra[i].value, 1, :)
                    bbox_indices = keep_loop ? view(model_indices, CartesianIndices(lbox), l) : view(model_indices, CartesianIndices(lbox))
                    for (k, idx) in enumerate(bbox_indices)
                        # e.g. Atomically accumulate into the flat `model_view`
                        Atomix.@atomic model_view[idx] += pr[k]
                    end

                    for k in eachindex(pr)
                    end
                catch e
                    @debug "Error on lenslet $i" exception = e
                    profiles[i] = nothing
                    lamp_spectra[i] = nothing
                end
                verbose && next!(progress)
            end
        end

        valid_lenslets = map(!isnothing, profiles)

    end
    verbose && ProgressMeter.finish!(progress)
    return profiles, lamp_spectra, model
end


function get_meanx(data::WeightedArray{T, N}, bbox; relative = false) where {T, N}
    if relative
        (; value, precision) = data
    else
        (; value, precision) = view(data, bbox)
    end
    ax, _ = axes(bbox)

    return sum(value .* sqrt.(precision) .* ax) ./ sum(sqrt.(precision) .* value)
end


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
    f(x) = likelihood(ScaledL2Loss(dims = 1, nonnegative = true), d, re(x)())
    Newuoa.optimize!(f, vec, 1, 1.0e-9; scale = scale, check = false, maxeval = maxeval, verbose = verbose)
    return re(vec)
end

build_loss(data, re) = x -> likelihood(ScaledL2Loss(dims = 1, nonnegative = true), data, re(x)())
