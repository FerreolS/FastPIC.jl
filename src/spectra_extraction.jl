"""
    extract_spectrum(data::WeightedArray, profile::Profile; restrict=0, nonnegative=false, inbbox=false)

Extract a 1D spectrum from 2D/3D data using optimal weighted extraction.

Performs weighted least-squares fitting of the profile model to data:
`α = (P^T W P)^(-1) P^T W d`

where P is the profile, W is the precision matrix, and d is the data.

# Arguments
- `data`: Input weighted data (2D or 3D)
- `profile`: Profile model for extraction
- `restrict`: Threshold for profile truncation (0 = no truncation)
- `nonnegative`: Enforce non-negative extracted values
- `inbbox`: If true, assumes data is already cropped to profile bbox

# Returns
- `WeightedArray{T}`: Extracted spectrum with uncertainties

# Examples
```julia
spectrum = extract_spectrum(detector_data, trace_profile; nonnegative=true)
```
"""
function extract_spectrum(
        data::WeightedArray{T, N},
        profile::Profile{T2, M};
        restrict = 0,
        nonnegative = false,
        inbbox = false
    ) where {T, N, T2, M}
    bbox = profile.bbox
    if inbbox
        (; value, precision) = data
    else
        if N > 2
            (; value, precision) = view(data, axes(profile.bbox, 1), axes(profile.bbox, 2), :)
        else
            (; value, precision) = view(data, bbox)
        end
    end
    model = profile()

    if restrict > 0
        model .*= (model .> T2(restrict))
    end

    αprecision = dropdims(sum(model .^ 2 .* precision, dims = 1), dims = 1)
    α = dropdims(sum(model .* precision .* value, dims = 1), dims = 1) ./ αprecision

    nanpix = .!isnan.(α)
    if nonnegative
        positive = nanpix .& (α .>= T(0))
    else
        positive = nanpix
    end

    return WeightedArray(positive .* α, positive .* αprecision)
end


"""
    extract_spectra(data::WeightedArray, profiles::Vector; restrict=0, nonnegative=false, ntasks=4*Threads.nthreads(), refinement_loop=0, extra_width=5)

Extract multiple spectra from data using an array of profile models.

Parallelized version of `extract_spectrum` for processing multiple traces simultaneously.

# Arguments
- `data`: Input weighted data (2D or 3D)
- `profiles`: Vector of valid `Profile` objects
- `restrict`: Profile truncation threshold
- `nonnegative`: Enforce non-negative extracted values
- `ntasks`: Number of parallel tasks for processing

# Returns
- `Vector{WeightedArray{T,1}}`: Array of extracted spectra
"""
is_spectrum(value) = value isa WeightedArray

function extract_spectra(
        data::WeightedArray{T, N},
        profiles;
        transmission = FastUniformArray(T(1), length(profiles)),
        restrict = 0,
        nonnegative::Bool = true,
        ntasks = 4 * Threads.nthreads()
    ) where {T <: Real, N}
    (1 < N <= 3) || error("extract_spectra: data must have 2 or 3 dimensions")
    if profiles isa AbstractVector{<:Profile}
        spectra = Vector{WeightedArray{T, N - 1}}(undef, length(profiles))
    else
        spectra = Vector{Union{WeightedArray{T, N - 1}, LensletError}}(undef, length(profiles))
    end
    tmap!(spectra, profiles; ntasks = ntasks) do profile
        if is_profile(profile)
            try
                output = extract_spectrum(data, profile; restrict = restrict, nonnegative = nonnegative)
            catch
                output = zeros(WeightedValue{T}, size(profile.bbox))
            end
        else
            output = profile
        end
    end

    if transmission isa FastUniformArray
        return spectra
    end

    if Base.typesplit(eltype(transmission), LensletError) <: Real
        @localize spectra  tforeach(findall(is_profile, profiles); ntasks = ntasks) do i
            if is_spectrum(spectra[i])
                spectra[i] = spectra[i] ./ T(transmission[i])
            end
        end
    else
        spectra = correct_spectral_transmission(spectra, transmission)
    end
    return spectra
end


#=
Variance estimation from
Díaz-Francés, Eloísa; Rubio, Francisco J. (2012-01-24). "On the existence of a normal approximation to the distribution of the ratio of two independent normal random variables". Statistical Papers
=#


function correct_spectral_transmission(
        spectra::AbstractVector{<:AbstractArray{<:WeightedValue{T}}},
        transmission
    ) where {T}
    corrected = similar(spectra)
    tforeach(axes(corrected, 1); ntasks = 4 * Threads.nthreads()) do i
        (; value, precision) = transmission[i]
        spec_val = get_value(spectra[i])
        spec_prec = spectra[i].precision
        mean_spec = @. T(spec_val / value)
        var_spec = @. T(inv(spec_prec) / (value .^ 2) + (spec_val / value) .^ 2 .* inv(precision) / (value .^ 2))
        corrected[i] = WeightedArray(mean_spec, inv.(var_spec))
    end
    return corrected
end


flatten_spectra(spectra::AbstractVector{<:Union{<:AbstractArray{<:WeightedValue}, LensletError}}) = error("flatten_spectra:  Profiles must be cleaned up from LensletError before extracting spectra for  flattening")

function flatten_spectra(spectra::AbstractVector{<:AbstractArray{<:WeightedValue, N}}) where {N}
    sp1 = first(spectra)
    T = eltype(get_value(sp1))
    if N == 1
        precision = zeros(T, length(spectra), length(sp1))
        value = zeros(T, length(spectra), length(sp1))
        foreach(enumerate(spectra)) do (i, sp)
            precision[i, :] .= sp.precision
            value[i, :] .= sp.value
        end
    else
        precision = zeros(T, length(spectra), size(sp1, 1), size(sp1, 2))
        value = zeros(T, length(spectra), size(sp1, 1), size(sp1, 2))
        foreach(enumerate(spectra)) do (i, sp)
            precision[i, :, :] .= sp.precision
            value[i, :, :] .= sp.value
        end
    end
    return WeightedArray(value, precision)
end


"""
    filter_spectra_outliers!(spectra; threshold=3)

Remove outliers from extracted spectra using robust statistics.

Identifies and zeros out spectral points that deviate more than `threshold` 
median absolute deviations from the median spectrum across all profiles.

# Arguments
- `spectra`: Vector of WeightedArray spectra (modified in-place)
- `threshold`: Outlier detection threshold in MAD units

Sets precision to 0 and value to 0 for detected outliers.
"""
function filter_spectra_outliers(spectra; kwargs...)
    spectra_copy = deepcopy(spectra)
    filter_spectra_outliers!(spectra_copy; kwargs...)
    return spectra_copy
end

function filter_spectra_outliers!(
        spectra;
        threshold = 3
    )
    valid_spectra = findall(is_spectrum, spectra)
    q = hcat([spectra[i].value for i in valid_spectra ]...)
    m = median(q; dims = 2) # median spectrum
    s = mad.(eachslice(q, dims = 1); normalize = true)
    bad = @. !((m - threshold * s) <= q <= (m + threshold * s))
    for i in eachindex(IndexCartesian(), bad)
        if bad[i]
            spectra[valid_spectra[i[2]]].precision[i[1]] = 0
            spectra[valid_spectra[i[2]]].value[i[1]] = 0
        end
    end

    return
end
