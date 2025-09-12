struct LaserModel
    #   wavelength::Vector{Float64}
    position::Vector{Float64}
    fwhm::Vector{Float64}
end
#trainable(x::LaserModel) = (; position=x.position, fwhm=x.fwhm)

LaserModel(position::AbstractVector, fwhm::AbstractVector) =
    LaserModel(collect(position), collect(fwhm))

function compute_laser_images((; position, fwhm)::LaserModel, idx::AbstractVector)
    fwhm2sigma = 1 / (2 * sqrt(2 * log(2)))
    fw = 2 .* (fwhm .* fwhm2sigma)
    return exp.(-((idx .- reshape(position, 1, :)) ./ reshape(fw, 1, :)) .^ 2)
end

function compute_lasers_amplitudes(
        ::Val{N},
        model::Array{T, 2}, (; value, precision)::WeightedArray
    ) where {T <: Real, N}

    A = @MMatrix zeros(T, N, N)
    b = @MVector zeros(T, N)

    @inbounds for index in 1:N
        mw = model[:, index] .* precision
        b[index] = mw' * value
        A[index, index] = mw' * model[:, index]
        for i in 1:(index - 1)
            A[i, index] = A[index, i] = mw' * model[:, i]
        end
    end

    return inv(A) * b

end

function laser_cost(
        data::WeightedArray,
        lasers::LaserModel
    )

    images = hcat(compute_laser_images(lasers, axes(data, 1)), ones(length(data)))
    amplitude = compute_lasers_amplitudes(Val(size(images, 2)), images, data)
    model = images * amplitude
    return likelihood(data, model)
end

function fit_laser(
        data::WeightedArray,
        laser::LaserModel
    )

    vec, re = Optimisers.destructure(laser)
    f(x) = laser_cost(data, re(x))
    Newuoa.optimize!(f, vec, 1.0e-5, 1.0e-15; check = false, maxeval = 10_000, verbose = 0)

    return re(vec)
end

get_laser_precision(laser::LaserModel, data::WeightedArray) = get_laser_precision(Val(length(laser.position)), laser, data)

function get_laser_precision(
        ::Val{N},
        laser::LaserModel,
        (; value, precision)::WeightedArray
    ) where {N}
    images = compute_laser_images(laser, axes(value, 1))
    W = @MMatrix zeros(Float64, N, N)

    @inbounds for index in 1:N
        mw = images[:, index] .* precision
        W[index, index] = mw' * images[:, index]
        for i in 1:(index - 1)
            W[i, index] = W[index, i] = mw' * images[:, i]
        end
    end
    return W
end

spectral_calibration(order, ref, lasers_λs, laser_positions, Wpos) = spectral_calibration(Val(order), Val(length(laser_positions)), ref, lasers_λs, laser_positions, Wpos)

function spectral_calibration(::Val{order}, ::Val{lines}, ref, lasers_λs, laser_positions, Wpos) where {order, lines}
    A = MMatrix{lines, order + 1}(((laser_positions .- ref) ./ ref) .^ reshape(0:order, 1, :))
    coefs = inv(A' * Wpos * A) * A' * Wpos * lasers_λs
    return coefs
end


function build_λrange(λs::AbstractMatrix{<:Real}; superres = 1)
    nb_el = round(Int, size(λs, 1) * superres)
    blue = λs[1, :]
    red = λs[end, :]
    return range(start = minimum(blue), stop = maximum(red), step = median((red .- blue)) ./ nb_el)
end

function build_λrange(λs::Vector{Vector{Float64}}, valid_lenslets; superres = 1)
    idx = findall(valid_lenslets)
    nb_el = round(Int, maximum(length.(λs[idx])) * superres)
    blue = minimum.(λs[idx])
    red = maximum.(λs[idx])
    return range(start = minimum(blue), stop = maximum(red), step = median((red .- blue)) ./ nb_el)
end

function estimate_template(λ, coefs, reference_pixel, spectra, valid_lenslets)
    nλ = length(λ)
    transmission = zeros(Float64, length(valid_lenslets))
    MI = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(valid_lenslets))
    A = zeros(Float64, nλ, nλ)
    diagA = 2 * ones(Float64, nλ)
    diagA[1] = 1
    diagA[end] = 1
    A = Array(BandedMatrix((0 => diagA, 1 => -1 * ones(nλ - 1), -1 => -1 * ones(nλ - 1)), (nλ, nλ)))
    # TO BE FIXED : ADD TIKHONOV PARAMETER AS INPUT
    b = zeros(Float64, nλ)
    foreach(findall(valid_lenslets)) do idx
        (; value, precision) = spectra[idx]

        profile_wavelength = get_wavelength(coefs[idx], reference_pixel, 1:length(value))
        MI[idx] = build_sparse_interpolation_integration_matrix(λ, get_lower_uppersamples(profile_wavelength)...)
        b .+= Array(MI[idx]' * (precision .* value))
        A .+= Array(MI[idx]' * (precision .* MI[idx]))
    end


    template = A \ b

    OhMyThreads.tforeach(findall(valid_lenslets)) do idx
        (; value, precision) = spectra[idx]
        m = (MI[idx] * template)
        transmission[idx] = sum((mp = m .* precision) .* value) / sum(m .* mp)
    end

    transmission .*= 1 ./ median(transmission[findall(valid_lenslets)])

    return template, transmission
end

function lamp_model(λ, spectrum_template, template_wavelength)
    lo, up = get_lower_uppersamples(λ)
    model = build_sparse_interpolation_integration_matrix(template_wavelength, lo, up) * spectrum_template
    return model
end

function laser_model(λ, fwhm_pixels, lasers_λs, data)
    idx = max.(2, [searchsortedlast(λ, l) for l in lasers_λs])
    las = LaserModel(lasers_λs, fwhm_pixels .* (λ[idx] .- λ[idx .- 1]))
    images = hcat(compute_laser_images(las, λ), ones(length(λ)))
    amplitude = compute_lasers_amplitudes(Val(length(lasers_λs) + 1), images, data)
    return images * amplitude
end


function spectral_refinement(coefs, lamp, lamp_template, wavelength, reference_pixel, lasers_λs, fwhm_pixels, laser)
    function loss(x)
        wvlngth = get_wavelength(x, reference_pixel, axes(lamp, 1))
        lamp_spectrum = lamp_model(wvlngth, lamp_template, wavelength)
        laser_spectrum = laser_model(wvlngth, fwhm_pixels, lasers_λs, laser)
        return likelihood(ScaledL2Loss(), lamp, lamp_spectrum) + likelihood(laser, laser_spectrum)
    end
    #scale = 1e-7 .* vcat(10. .^ (-(1:length(coefs))))
    scale = 1.0e-8 .* ones(length(coefs))
    Newuoa.optimize!(loss, coefs, 1, 1.0e-9; scale = scale, check = false, maxeval = 10_000, verbose = 0)
    return coefs
end
function recalibrate_wavelengths(
        λ,
        coefs,
        order,
        lamp_profile,
        laser_profile,
        lasers_λs,
        lasers_model,
        reference_pixel,
        valid_lenslets;
        loop = 2 # TODO put in calib_params
    )

    template, transmission = estimate_template(λ, coefs, reference_pixel, lamp_profile, valid_lenslets)

    new_coefs = similar(coefs)

    p = Progress(sum(valid_lenslets) * loop; showspeed = true)

    for _ in 1:loop
        @localize template @localize coefs OhMyThreads.tforeach(findall(valid_lenslets)) do i
            if (order + 1) > length(coefs[i])
                coef = vcat(coefs[i], zeros(order - length(coefs[i]) + 1))
            else
                coef = copy(coefs[i])
            end
            try
                new_coefs[i] = spectral_refinement(coef, lamp_profile[i], template, λ, reference_pixel, lasers_λs, lasers_model[i].fwhm, laser_profile[i])
            catch e
                @debug "Spectral refinement failed for lenslet $i: $e"
                valid_lenslets[i] = false
            end
            next!(p)
        end
        coefs = copy(new_coefs)

        template, transmission = estimate_template(λ, coefs, reference_pixel, lamp_profile, valid_lenslets)

    end
    ProgressMeter.finish!(p)
    return coefs, template, transmission
end


function spectral_calibration(
        lasers,
        lamp_spectra,
        profiles;
        valid_lenslets = trues(length(profiles)),
        calib_params::FastPICParams = FastPICParams(),
        loop = 2,
        superres = 1,
        final_spectral_order = 3
    )

    @unpack_FastPICParams calib_params
    @unpack_BboxParams bbox_params

    profile_type = ZippedVector{WeightedValue{Float64}, 2, true, Tuple{Vector{Float64}, Vector{Float64}}}
    laser_profile = Vector{profile_type}(undef, NLENS)
    laser_model = LaserModel([7.0, 20.0, 35.0], [2.0, 2.0, 2.0])
    coefs = Vector{Vector{Float64}}(undef, NLENS)
    λ = Vector{Vector{Float64}}(undef, NLENS)
    las = Vector{typeof(laser_model)}(undef, NLENS)

    #Threads.@threads for i in findall(valid_lenslets)
    # from https://discourse.julialang.org/t/optionally-multi-threaded-for-loop/81902/8?u=skleinbo
    _foreach = multi_thread ? (@localize coefs OhMyThreads.tforeach) : Base.foreach
    progress = Progress(sum(valid_lenslets); showspeed = true)
    @localize coefs _foreach(findall(valid_lenslets)) do i
        if sum(view(lasers, profiles[i].bbox).precision) == 0
            valid_lenslets[i] = false
        else
            try
                laser_profile[i] = extract_spectrum(lasers, profiles[i])
                las[i] = fit_laser(laser_profile[i], laser_model)

                if std(las[i].position .- laser_model.position) > 1
                    throw("Laser position too far from initial guess for lenslet $i")
                end
                W = get_laser_precision(las[i], laser_profile[i])
                if any(diag(W) .< 1.0e-4)
                    throw("W singular  for lenslet $i")
                end
                coefs[i] = spectral_calibration(
                    spectral_initial_order
                    , reference_pixel, lasers_λs, las[i].position, W
                )
                if any(isnan.(coefs[i]))
                    throw("NaN found in coefs for lenslet $i")
                end
                λ[i] = get_wavelength(coefs[i], reference_pixel, axes(laser_profile[i], 1))

            catch e
                @debug "Error on lenslet $i" exception = e
                valid_lenslets[i] = false
            end
        end
        next!(progress)
    end
    finish!(progress)
    lλ = build_λrange(λ, valid_lenslets; superres = superres)

    coefs, template, transmission = recalibrate_wavelengths(
        lλ,
        coefs,
        final_spectral_order,
        lamp_spectra,
        laser_profile,
        lasers_λs,
        las,
        reference_pixel,
        valid_lenslets;
        loop = loop
    )
    return coefs, template, transmission, lλ, las, laser_profile, valid_lenslets
end
