function estimate_transmission(
        profiles::Vector{<:Profile},
        lamp::WeightedArray{T},
        λ::AbstractVector,
        template::AbstractVector;
        transmission_threshold = 0.5
    ) where {T}

    length(template) == length(λ) || error("template and λ must have the same length")
    lamp_spectra = extract_spectra(lamp, profiles; restrict = 0, nonnegative = true)
    data = flatten_spectra(lamp_spectra)

    P = build_LinOpIntegration_operators(profiles, λ; T = T)
    flatcube = ones(T, size(data, 1)) .* reshape(T.(template), 1, :)

    flatspectra = P * flatcube

    transmission = data ./ flatspectra
    transmission ./= median(transmission.value)
    goodpix = transmission_threshold .< (transmission.value) .< (1 + transmission_threshold)
    (; value, precision) = transmission
    value[.!goodpix] .= T(1)
    precision[.!goodpix] .= T(0)
    transmission = WeightedArray(value, precision)
    unflatten_transmission = [transmission[i, :] for i in 1:size(transmission, 1)]

    return unflatten_transmission
end
