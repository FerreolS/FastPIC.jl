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
    extract_spectra(data::WeightedArray, profiles::Vector; restrict=0, nonnegative=false, ntasks=4*Threads.nthreads())

Extract multiple spectra from data using an array of profile models.

Parallelized version of `extract_spectrum` for processing multiple traces simultaneously.

# Arguments
- `data`: Input weighted data (2D or 3D)
- `profiles`: Vector of Profile objects (Nothing for invalid traces)
- `restrict`: Profile truncation threshold
- `nonnegative`: Enforce non-negative extracted values
- `ntasks`: Number of parallel tasks for processing

# Returns
- `Vector{Union{WeightedArray{T,1}, Nothing}}`: Array of extracted spectra
"""
function extract_spectra(
        data::WeightedArray{T, N},
        profiles::Vector{Union{Profile{T2, M}, Nothing}};
        transmission = FastUniformArray(T2(1), length(profiles)),
        restrict = 0,
        nonnegative::Bool = true,
        ntasks = 4 * Threads.nthreads(),
        refinement_loop = 0,
        extra_width = 5
    ) where {T <: Real, N, T2 <: Real, M}
    (1 < N <= 3) || error("extract_spectra: data must have 2 or 3 dimensions")
    profile_type = ZippedVector{WeightedValue{T2}, 2, true, Tuple{Array{T2, N - 1}, Array{T2, N - 1}}}
    spectra = Vector{Union{profile_type, Nothing}}(undef, length(profiles))
    fill!(spectra, nothing)

    if refinement_loop > 0
        if N == 3
            for t in axes(data, 3)
                _, spctr, _ = refine_lamp_model(view(data, :, :, t), profiles; keep_loop = false, profile_loop = refinement_loop, verbose = false, extra_width = extra_width, lamp_extract_restrict = restrict, dont_fit_profile = true)
                foreach(findall(!isnothing, profiles)) do i
                    spectra[i, t] .= spctr[i]
                end
            end
        else
            _, spectra, _ = refine_lamp_model(data, profiles; keep_loop = false, profile_loop = refinement_loop, verbose = false, extra_width = extra_width, lamp_extract_restrict = restrict, dont_fit_profile = true)
        end
    else
        tforeach(findall(!isnothing, profiles); ntasks = ntasks) do i
            spectra[i] = extract_spectrum(data, profiles[i]; restrict = restrict, nonnegative = nonnegative) ./ T2(transmission[i])
        end
    end
    return spectra
end

"""
    estimate_shift(
        data::WeightedArray,
        profiles::Vector{<:Union{Nothing, Profile}},
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
        profiles::Vector{<:Union{Nothing, Profile}},
        profile_wavelength::Vector{<:Union{Nothing, AbstractVector{Float64}}},
        transmission::Vector{Float64},
        template::Vector{Float64},
        λ::AbstractVector{Float64};
        ntasks = 4 * Threads.nthreads(),
        restrict = 0,
    ) where {T <: Real, N}


    models = Vector{Union{Nothing, Vector{Float64}}}(undef, length(profiles))
    fill!(models, nothing)

    tforeach(findall(!isnothing, profile_wavelength); ntasks = ntasks) do i
        MI = build_sparse_interpolation_integration_matrix(λ, get_lower_uppersamples(profile_wavelength[i])...)
        models[i] = (MI * template) .* transmission[i]

    end


    loss(shift) = tmapreduce(+, findall(!isnothing, profiles); outputtype = Float64, ntasks = ntasks) do idx
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
