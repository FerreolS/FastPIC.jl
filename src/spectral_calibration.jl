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

laser_calibration(order, ref, lasers_λs, laser_positions, Wpos) = laser_calibration(Val(order), Val(length(laser_positions)), ref, lasers_λs, laser_positions, Wpos)

function laser_calibration(::Val{order}, ::Val{lines}, ref, lasers_λs, laser_positions, Wpos) where {order, lines}
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

function estimate_template(λ, coefs, reference_pixel, spectra, valid_lenslets; regul = 1)
    nλ = length(λ)
    transmission = zeros(Float64, length(valid_lenslets))
    MI = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(valid_lenslets))
    if regul == 0
        A = zeros(Float64, nλ, nλ)
    else
        diagA = 2 * regul * ones(Float64, nλ)
        diagA[1] = regul
        diagA[end] = regul
        A = Array(BandedMatrix((0 => diagA, 1 => -regul * ones(nλ - 1), -1 => -regul * ones(nλ - 1)), (nλ, nλ)))
    end
    # TO BE FIXED : ADD TIKHONOV PARAMETER AS INPUT
    b = zeros(Float64, nλ)
    foreach(findall(valid_lenslets)) do idx
        (; value, precision) = spectra[idx]
        profile_wavelength = get_wavelength(coefs[idx], reference_pixel, axes(value, 1))
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
        lamp_spectra,
        laser_spectra,
        lasers_λs,
        lasers_model,
        reference_pixel,
        valid_lenslets;
        verbose = false,
        ntasks = Threads.nthreads() * 4,
        regul = 1,
        loop = 2 # TODO put in calib_params
    )

    template, transmission = estimate_template(λ, coefs, reference_pixel, lamp_spectra, valid_lenslets; regul = regul)

    new_coefs = similar(coefs)

    progressbar = verbose ? Progress(sum(valid_lenslets) * loop; showspeed = true, desc = "Spectral recalibration $loop loops") : nothing

    for _ in 1:loop
        @localize progressbar @localize template @localize coefs OhMyThreads.tforeach(findall(valid_lenslets); ntasks = ntasks) do i
            #foreach(findall(valid_lenslets)) do i
            if (order + 1) > length(coefs[i])
                coef = vcat(coefs[i], zeros(order - length(coefs[i]) + 1))
            else
                coef = copy(coefs[i])
            end
            try
                new_coefs[i] = spectral_refinement(coef, lamp_spectra[i], template, λ, reference_pixel, lasers_λs, lasers_model[i].fwhm, laser_spectra[i])
            catch e
                @debug "Spectral refinement failed for lenslet $i: $e"
                valid_lenslets[i] = false
            end
            verbose && next!(progressbar)
        end
        coefs = copy(new_coefs)

        template, transmission = estimate_template(λ, coefs, reference_pixel, lamp_spectra, valid_lenslets)

    end
    verbose && (finish!(progressbar))
    return coefs, template, transmission
end

function spectral_calibration(
        lasers::WeightedArray{T, 2},
        lamp_spectra::Vector{L},
        profiles::AbstractVector{<:Union{Nothing, Profile}};
        calib_params::FastPICParams = FastPICParams()
    ) where {T, L <: Union{Nothing, WeightedArray{T, 1}}}

    @unpack_FastPICParams calib_params

    laser_spectra, las, coefs, λs, valid_lenslets = laser_calibration!(L, lasers, profiles; calib_params = calib_params)
    lλ = build_λrange(λs, valid_lenslets; superres = spectral_superres)

    coefs, template, transmission = recalibrate_wavelengths(
        lλ,
        coefs,
        spectral_final_order,
        lamp_spectra,
        laser_spectra,
        lasers_λs,
        las,
        reference_pixel,
        valid_lenslets;
        regul = spectral_recalibration_regul,
        loop = spectral_recalibration_loop,
        ntasks = ntasks,
        verbose = spectral_calibration_verbose
    )
    return coefs, template, transmission, lλ,   valid_lenslets
end


function laser_calibration!(
        ::Type{L}, lasers,
        profiles; calib_params::FastPICParams = FastPICParams()
    ) where {L}
    @unpack_FastPICParams calib_params

    laser_spectra = Vector{L}(undef, NLENS)
    laser_model = LaserModel([7.0, 20.0, 35.0], [2.0, 2.0, 2.0])
    coefs = Vector{Vector{Float64}}(undef, NLENS)
    λs = Vector{Vector{Float64}}(undef, NLENS)
    las = Vector{typeof(laser_model)}(undef, NLENS)

    valid_lenslets = map(!isnothing, profiles)
    progressbar = spectral_calibration_verbose ? Progress(sum(valid_lenslets); showspeed = true, desc = "Spectral calibration") : nothing

    @localize profiles @localize laser_spectra @localize coefs OhMyThreads.tforeach(findall(valid_lenslets); ntasks = ntasks) do i
        if sum(view(lasers, profiles[i].bbox).precision) == 0
            valid_lenslets[i] = false
        else
            try
                laser_spectra[i] = extract_spectrum(lasers, profiles[i]; restrict = laser_extract_restrict)
                las[i] = fit_laser(laser_spectra[i], laser_model)

                if std(las[i].position .- laser_model.position) > 1
                    throw("Laser position too far from initial guess for lenslet $i")
                end
                W = get_laser_precision(las[i], laser_spectra[i])
                if any(diag(W) .< 1.0e-4)
                    throw("W singular  for lenslet $i")
                end
                coefs[i] = laser_calibration(
                    spectral_initial_order, reference_pixel, lasers_λs, las[i].position, W
                )
                if any(isnan.(coefs[i]))
                    throw("NaN found in coefs for lenslet $i")
                end
                λs[i] = get_wavelength(coefs[i], reference_pixel, axes(laser_spectra[i], 1))

            catch e
                @debug "Error on lenslet $i" exception = (e, catch_backtrace())
                valid_lenslets[i] = false
                profiles[i] = nothing
            end
        end
        spectral_calibration_verbose && next!(progressbar)
    end
    spectral_calibration_verbose && finish!(progressbar)
    return laser_spectra, las, coefs, λs, valid_lenslets
end
