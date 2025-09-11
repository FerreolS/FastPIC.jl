
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

function compute_lasers_amplitudes(::Val{N},
    model::Array{T,2}, (; value, precision)::WeightedArray
) where {T<:Real,N}

    A = @MMatrix zeros(T, N, N)
    b = @MVector zeros(T, N)

    @inbounds for index = 1:N
        mw = model[:, index] .* precision
        b[index] = mw' * value
        A[index, index] = mw' * model[:, index]
        for i = 1:index-1
            A[i, index] = A[index, i] = mw' * model[:, i]
        end
    end

    return inv(A) * b

end

function laser_cost(data::WeightedArray,
    lasers::LaserModel
)

    images = hcat(compute_laser_images(lasers, axes(data, 1)), ones(length(data)))
    amplitude = compute_lasers_amplitudes(Val(size(images, 2)), images, data)
    model = images * amplitude
    return likelihood(data, model)
end

function fit_laser(data::WeightedArray,
    laser::LaserModel)

    vec, re = Optimisers.destructure(laser)
    f(x) = laser_cost(data, re(x))
    Newuoa.optimize!(f, vec, 1e-5, 1e-15; check=false, maxeval=10_000, verbose=0)

    return re(vec)
end

get_laser_precision(laser::LaserModel, data::WeightedArray) = get_laser_precision(Val(length(laser.position)), laser, data)

function get_laser_precision(::Val{N},
    laser::LaserModel,
    (; value, precision)::WeightedArray) where {N}
    images = compute_laser_images(laser, axes(value, 1))
    W = @MMatrix zeros(Float64, N, N)

    @inbounds for index = 1:N
        mw = images[:, index] .* precision
        W[index, index] = mw' * images[:, index]
        for i = 1:index-1
            W[i, index] = W[index, i] = mw' * images[:, i]
        end
    end
    return W
end

spectral_calibration(order, ref, lasers_λs, laser_positions, Wpos) = spectral_calibration(Val(order), Val(length(laser_positions)), ref, lasers_λs, laser_positions, Wpos)

function spectral_calibration(::Val{order}, ::Val{lines}, ref, lasers_λs, laser_positions, Wpos) where {order,lines}
    A = MMatrix{lines,order + 1}(((laser_positions .- ref) ./ ref) .^ reshape(0:order, 1, :))
    coefs = inv(A' * Wpos * A) * A' * Wpos * lasers_λs
    return coefs
end


function spectral_calibration(
    lasers,
    lamp_spectra,
    profiles;
    valid_lenslets=trues(length(profiles)),
    calib_params::FastPICParams=FastPICParams(),
    loop=2,
    superres=1,
    final_spectral_order=3
)

    @unpack_FastPICParams calib_params
    @unpack_BboxParams bbox_params

    profile_type = ZippedVector{WeightedValue{Float64},2,true,Tuple{Vector{Float64},Vector{Float64}}}
    laser_profile = Vector{profile_type}(undef, NLENS)
    laser_model = LaserModel([7.0, 20.0, 35.0], [2.0, 2.0, 2.0])
    coefs = Vector{Vector{Float64}}(undef, NLENS)
    λ = Vector{Vector{Float64}}(undef, NLENS)
    las = Vector{typeof(laser_model)}(undef, NLENS)

    #Threads.@threads for i in findall(valid_lenslets)
    # from https://discourse.julialang.org/t/optionally-multi-threaded-for-loop/81902/8?u=skleinbo
    _foreach = multi_thread ? (@localize coefs OhMyThreads.tforeach) : Base.foreach
    progress = Progress(sum(valid_lenslets); showspeed=true)
    @localize coefs _foreach(findall(valid_lenslets)) do i
        if sum(view(lasers, profiles[i].bbox).precision) == 0
            valid_lenslets[i] = false
        else
            try
                laser_profile[i] = extract_model(lasers, profiles[i])
                las[i] = fit_laser(laser_profile[i], laser_model)

                if std(las[i].position .- laser_model.position) > 1
                    throw("Laser position too far from initial guess for lenslet $i")
                end
                W = get_laser_precision(las[i], laser_profile[i])
                if any(diag(W) .< 1e-4)
                    throw("W singular  for lenslet $i")
                end
                coefs[i] = spectral_calibration(spectral_order, reference_pixel, lasers_λs, las[i].position, W)
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
    lλ = build_λrange(λ, valid_lenslets; superres=superres)

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
        loop=loop)
    return coefs, template, transmission, lλ, las, laser_profile, valid_lenslets
end