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
            (; value, precision) = view(data, bbox, :)
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
        profiles::AbstractVector{<:Profile};
        transmission = FastUniformArray(T(1), length(profiles)),
        restrict = 0,
        nonnegative::Bool = true,
        ntasks = 4 * Threads.nthreads(),
        refinement_loop = 0,
        extra_width = 5
    ) where {T <: Real, N}
    (1 < N <= 3) || error("extract_spectra: data must have 2 or 3 dimensions")
    profile_type = ZippedArray{WeightedValue{T}, N - 1, 2, true, Tuple{Array{T, N - 1}, Array{T, N - 1}}}
    spectra = Vector{profile_type}(undef, length(profiles))


    if refinement_loop > 0
        if N == 3
            nframes = size(data, 3)
            foreach(findall(is_profile, profiles)) do i
                ny = size(profiles[i].bbox, 2)
                spectra[i] = WeightedArray(zeros(T, ny, nframes), zeros(T, ny, nframes))
            end
            for t in axes(data, 3)
                #   tforeach(axes(data, 3); ntasks = ntasks) do t
                _, spctr, _ = refine_lamp_model(view(data, :, :, t), profiles; keep_loop = false, profile_loop = refinement_loop, verbose = false, extra_width = extra_width, lamp_extract_restrict = restrict, dont_fit_profile = true)
                foreach(findall(is_profile, profiles)) do i
                    spectra[i].value[:, t] .= spctr[i].value
                    spectra[i].precision[:, t] .= spctr[i].precision
                end
            end
        else
            _, spectra, _ = refine_lamp_model(data, profiles; keep_loop = false, profile_loop = refinement_loop, verbose = false, extra_width = extra_width, lamp_extract_restrict = restrict, dont_fit_profile = true)
        end
    else
        @localize spectra tforeach(findall(is_profile, profiles); ntasks = ntasks) do i
            try
                spectra[i] = extract_spectrum(data, profiles[i]; restrict = restrict, nonnegative = nonnegative)
            catch
                spectra[i] = WeightedArray(zeros(T, size(profiles[i].bbox)), zeros(T, size(profiles[i].bbox)))
            end
        end
    end
    if transmission isa FastUniformArray
        return spectra
    end

    if Base.typesplit(eltype(transmission), Nothing) <: Real
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

"""
    estimate_shift(
        data::WeightedArray,
        profiles::AbstractVector{<:Union{Profile, LensletError}},
        profile_wavelength::Vector{<:Union{Nothing, AbstractVector{Float64}}},
        transmission::Vector{Float64},
        template::Vector{Float64},
        λ::AbstractVector{Float64};
        ntasks = 4 * Threads.nthreads(),
        restrict = 0,
    ) -> Float64

Estimates a global lateral pixel shift by maximizing the cross-correlation between extracted spectra and a given template.

This function computes an optimal shift in the X direction (perpendicular to the profile) that best aligns the observed spectra with a template spectrum. It works by defining a `loss` function (which is actually a score to be maximized) that, for a given shift, extracts all spectra, and computes a weighted sum of their cross-correlation with the template. The `OptimPackNextGen.BraDi.maximize` routine is then used to find the shift that maximizes this score.

# Arguments
- `data::WeightedArray`: The input 2D/3D data cube with associated weights (precision).
- `profiles::Vector`: A vector of `Profile` objects, each defining the spatial trace for a spectrum to be extracted.
- `profile_wavelength::Vector`: A vector where each element is the wavelength solution (grid) corresponding to a profile.
- `transmission::Vector{Float64}`: A vector of transmission/throughput values, one for each profile.
- `template::Vector{Float64}`: The template spectrum to which the extracted spectra will be compared.
- `λ::AbstractVector{Float64}`: The wavelength grid corresponding to the `template` spectrum.

# Keyword Arguments
- `ntasks::Int`: The number of parallel tasks to use for computation. Defaults to `4 * Threads.nthreads()`.
- `restrict::Int`: An integer to restrict the extraction region, passed to `extract_spectrum`. Defaults to `0` (no restriction).

# Returns
- `Float64`: The estimated optimal shift in pixels.

# Method
1.  For each profile, a model spectrum is generated by interpolating the `template` onto the profile's specific wavelength grid and multiplying by its `transmission`.
2.  An objective function (`loss`) is defined. For a given `shift`:
    a. It applies the `shift` to the spatial center of each profile.
    b. It extracts the spectrum for each shifted profile using `extract_spectrum`.
    c. It computes a precision-weighted cross-correlation score between the extracted spectrum and its corresponding model.
    d. These scores are summed across all profiles.
3.  The `OptimPackNextGen.BraDi.maximize` algorithm is used to find the `shift` that maximizes the objective function, using `[-0.5, 0.0, 0.5]` as initial bracket points.
"""

function estimate_shift(
        data::WeightedArray{T, N},
        profiles::AbstractVector{<:Union{Profile, LensletError}},
        models::Vector{<:Union{Nothing, Vector{Float64}}};
        ntasks = 4 * Threads.nthreads(),
        restrict = 0,
    ) where {T <: Real, N}


    loss(shift) = tmapreduce(+, findall(is_profile, profiles); outputtype = Float64, ntasks = ntasks) do idx
        tmprofile = deepcopy(profiles[idx])
        tmprofile.cx[1] += shift
        (; value, precision) = extract_spectrum(data, tmprofile; restrict = restrict, nonnegative = true)
        mp = models[idx] .* precision
        denom = sum(mp .* models[idx])
        (denom == 0 || any(isnan, mp)) ? 0.0 : sum(mp .* value) / denom

    end

    shift = OptimPackNextGen.BraDi.maximize(loss, [-0.5, 0.0, 0.5])
    return shift[1]
end

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
