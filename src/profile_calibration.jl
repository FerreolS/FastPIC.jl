function calibrate_profile(
        lamp,
        ; calib_params::FastPICParams = FastPICParams(),
        valid_lenslets::AbstractVector{Bool} = trues(calib_params.NLENS)
    )


    @unpack_FastPICParams calib_params
    @unpack_BboxParams bbox_params

    size(valid_lenslets) == (NLENS,) || throw(ArgumentError("valid_lenslets must be of size NLENS"))


    bboxes, profiles = initialize_profile!(
        valid_lenslets,
        lamp;
        calib_params = calib_params
    )

    profile_type = ZippedVector{WeightedValue{Float64}, 2, true, Tuple{Vector{Float64}, Vector{Float64}}}
    lamp_spectra = Vector{profile_type}(undef, NLENS)

    progress = Progress(sum(valid_lenslets); showspeed = true)
    #Threads.@threads for i in findall(valid_lenslets)
    # from https://discourse.julialang.org/t/optionally-multi-threaded-for-loop/81902/8?u=skleinbo
    _foreach = multi_thread ? OhMyThreads.tforeach : Base.foreach
    @allow_boxed_captures _foreach(findall(valid_lenslets)) do i
        if sum(view(lamp, bboxes[i]).precision) == 0
            valid_lenslets[i] = false
        else
            try

                profiles[i] = fit_profile(lamp, profiles[i])
                if any(isnan.(profiles[i].cfwhm))
                    throw("NaN found in profile for lenslet $i")
                end
                lamp_spectra[i] = extract_model(lamp, profiles[i])

            catch e
                @debug "Error on lenslet $i" exception = (e, catch_backtrace())
                valid_lenslets[i] = false
            end
        end
        next!(progress)
    end
    ProgressMeter.finish!(progress)

    model = zeros(Float64, size(lamp))

    progress = Progress(sum(valid_lenslets) .* profile_loop; showspeed = true)

    for _ in 1:profile_loop
        res = lamp .- model
        model = zeros(Float64, size(lamp))
        for i in findall(valid_lenslets)
            resi = WeightedArray(view(res, profiles[i].bbox).value .+ profiles[i]() .* reshape(lamp_spectra[i].value, 1, :), view(res, profiles[i].bbox).precision)
            # resi = view(res,profiles[i].bbox) .+ profiles[i]() .* reshape(lamp_spectra[i].value, 1, :)

            profiles[i] = fit_profile(resi, profiles[i]; relative = true)
            if any(isnan.(profiles[i].cfwhm))
                valid_lenslets[i] = false
                continue
            end
            lamp_spectra[i] = extract_model(resi, profiles[i]; relative = true)
            if any(isnan.(lamp_spectra[i]))
                valid_lenslets[i] = false
                continue
            end
            (; xmin, xmax, ymin, ymax) = profiles[i].bbox
            lbox = BoundingBox(xmin = xmin - extra_width, xmax = xmax + extra_width, ymin = ymin, ymax = ymax)
            p = profiles[i](lbox)
            view(model, lbox) .+= p .* reshape(lamp_spectra[i].value, 1, :)
            next!(progress)
        end
    end
    ProgressMeter.finish!(progress)

    return profiles, bboxes, valid_lenslets, lamp_spectra, model
end

function initialize_profile!(
        valid_lenslets,
        lamp;
        calib_params::FastPICParams = FastPICParams()
    )

    @unpack_FastPICParams calib_params
    @unpack_BboxParams bbox_params


    bboxes = fill(BoundingBox{Int}(nothing), NLENS)
    profiles = Vector{Profile}(undef, NLENS)


    @inbounds for i in findall(valid_lenslets)
        bbox = get_bbox(lasers_cxy0s_init[i, 1], lasers_cxy0s_init[i, 2]; bbox_params = bbox_params)
        if ismissing(bbox)
            valid_lenslets[i] = false
        else
            bboxes[i] = bbox
            profiles[i] = Profile(bbox, lamp_cfwhms_init, vcat(get_meanx(lamp, bbox), zeros(profile_order)))
        end
    end
    return bboxes, profiles
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
        profile::Profile{M};
        relative = false
    ) where {T, N, M}

    fwhmorder = size(profile.cfwhm, 1)
    cxorder = length(profile.cx)
    if M == 1
        scale = vcat(10.0 .^ (-(1:(fwhmorder))), 10.0 .^ (-(1:(cxorder))))
    else
        scale = vcat(10.0 .^ (-(1:(fwhmorder))), 10.0 .^ (-(1:(cxorder))), 10.0 .^ (-(1:(cxorder))))
    end

    vec, re = Optimisers.destructure(profile)

    d = relative ? data : view(data, profile.bbox)
    f(x) = likelihood(ScaledL2Loss(dims = 1, nonnegative = true), d, re(x)())
    Newuoa.optimize!(f, vec, 1, 1.0e-9; scale = scale, check = false, maxeval = 10_000, verbose = 0)
    return re(vec)
end

build_loss(data, re) = x -> likelihood(ScaledL2Loss(dims = 1, nonnegative = true), data, re(x)())
