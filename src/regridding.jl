function spectral_regridding(
        data_spectra::Vector{<:Union{WeightedArray{T, 1}, Nothing}},
        profile_wavelength::Vector{<:Union{AbstractVector{Float64}, Nothing}},
        λ_grid::AbstractVector{Float64};
        ntasks = 4 * Threads.nthreads(),
    ) where {T <: Real}

    regridded_spectra = Vector{Union{WeightedArray{Float64, 1}, Nothing}}(undef, length(data_spectra))
    fill!(regridded_spectra, nothing)
    #    @localize regridded_spectra tforeach(findall(!isnothing, data_spectra); ntasks = ntasks) do i
    foreach(findall(!isnothing, data_spectra)) do i
        #  if !isnothing(profile_wavelength[i])
        widx = findall(x -> x > 0, data_spectra[i].precision)
        if length(widx) > 1
            itp_value = extrapolate(interpolate((profile_wavelength[i][widx],), data_spectra[i].value[widx], Gridded(Linear())), T(0.0))
            itp_prec = extrapolate(interpolate((profile_wavelength[i],), data_spectra[i].precision, Gridded(Linear())), T(0.0))
            regridded_spectra[i] = WeightedArray(itp_value.(λ_grid), max.(T(0.0), itp_prec.(λ_grid)))
        end
        #     end
    end
    return regridded_spectra
end

using ScatteredInterpolation
function spatial_regridding(
        spectral_reggridded_spectra::Vector{<:Union{WeightedArray{T, 1}, Nothing}},
        profiles::Vector{<:Union{Profile{T2, M}, Nothing}},
        n;
        ntasks = 4 * Threads.nthreads(),
    ) where {T <: Real, T2 <: Real, M}

    idx = findall(!isnothing, spectral_reggridded_spectra)
    spectral_len = length(spectral_reggridded_spectra[idx[1]])
    spectra_values = Matrix{T}(undef, spectral_len, length(idx))
    spectra_precision = Matrix{T}(undef, spectral_len, length(idx))
    px = zeros(Float64, length(idx))
    py = zeros(Float64, length(idx))
    fill!(spectra_values, T(0))
    fill!(spectra_precision, T(0))
    foreach(enumerate(idx)) do (i, v)
        spectra_values[:, i] = get_value(spectral_reggridded_spectra[v])
        spectra_precision[:, i] = spectral_reggridded_spectra[v].precision
        px[i] = profiles[v].cx[1]
        py[i] = profiles[v].ycenter
    end

    #regridded_values = zeros(T, length(x_grid), length(y_grid), spectral_len)
    # regridded_precision = zeros(T, length(x_grid), length(y_grid), spectral_len)
    #return spectra_values, spectra_precision, px, py

    Δ = 2048 / n
    points = vcat(px' ./ Δ, py' ./ Δ)

    X = repeat(1:n, n)[:]
    Y = repeat((1:n)', n)[:]
    gridPoints = [X Y]'
    gridded_value = zeros(Float64, n, n, spectral_len)
    gridded_precision = zeros(Float64, n, n, spectral_len)

    for l in axes(gridded_value, 3)
        itp = ScatteredInterpolation.interpolate(Multiquadratic(), points, spectra_values[l, :])
        interpolated = evaluate(itp, gridPoints)
        gridded_value[:, :, l] .= reshape(interpolated, n, n)

        itp = ScatteredInterpolation.interpolate(Multiquadratic(), points, spectra_precision[l, :])
        interpolated = evaluate(itp, gridPoints)
        gridded_precision[:, :, l] .= reshape(interpolated, n, n)
    end
    return WeightedArray(gridded_value, max.(T(0.0), gridded_precision))


    @localize regridded_values @localize regridded_precision tforeach(1:spectral_len; ntasks = ntasks) do l
        itp_value = extrapolate(interpolate((px, py), spectra_values[:, l], Gridded(Linear())), T(0.0))
        itp_prec = extrapolate(interpolate((px, py), spectra_precision[:, l], Gridded(Linear())), T(0.0))
        regridded_values[:, :, l] = itp_value.(x_grid, y_grid)
        regridded_precision[:, :, l] = max.(T(0.0), itp_prec.(x_grid, y_grid))
    end

    return WeightedArray(regridded_values, regridded_precision)


end
