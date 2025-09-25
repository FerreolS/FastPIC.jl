"""
    Profile{T,N}

A parametric model of the spectrum of each SPHERE/IFS lenslet.

# Fields
- `type::Type{T}`: Numeric type for computations (e.g., Float64)
- `bbox::BoundingBox{Int64}`: Bounding box defining the spatial extent on the lenslet on the detector
- `ycenter::Float64`: Central position along the dispersion axis
- `cfwhm::Array{Float64,N}`: Polynomial coefficients for FWHM variation (N=1 symmetric, N=2 asymmetric)
- `cx::Vector{Float64}`: Polynomial coefficients for lateral center position variation

# Examples
```julia
bbox = BoundingBox(xmin=10, xmax=50, ymin=100, ymax=150)
cfwhm = [2.5, 0.1]  # FWHM coefficients
cx = [25.0, 0.05]   # Center position coefficients
profile = Profile(Float64, bbox, 125.0, cfwhm, cx)
```
"""

struct Profile{T, N}
    type::Type{T}
    bbox::BoundingBox{Int64}
    ycenter::Float64
    cfwhm::Array{Float64, N}
    cx::Vector{Float64}
    function Profile(T::Type, bbox::BoundingBox{Int}, ycenter::Real, cfwhm::AbstractArray{T2, N}, cx::AbstractVector) where {N, T2 <: Real}
        @assert T <: Real
        size(cfwhm, 1) ≥ 1 || throw(ArgumentError("cfwhm must have at least one row"))
        length(cx) ≥ 1 || throw(ArgumentError("cx must have at least one element"))
        return new{T, N}(T, bbox, ycenter, collect(cfwhm), collect(cx))
    end
end

Profile(bbox::BoundingBox{Int}, cfwhm::AbstractArray, cx::AbstractVector) =
    Profile(bbox, mean(axes(bbox, 2)), cfwhm, cx)

Profile(T::Type, bbox::BoundingBox{Int}, cfwhm::AbstractArray, cx::AbstractVector) =
    Profile(T, bbox, mean(axes(bbox, 2)), cfwhm, cx)

Profile(bbox, ycenter, cfwhm, cx) = Profile(Float64, bbox, ycenter, cfwhm, cx)


((; type, bbox, ycenter, cfwhm, cx)::Profile)(; normalize = true) =
    get_profile(normalize ? Val(:normalize) : Val(:raw), type, bbox, ycenter, cfwhm, cx)
((; type, bbox, ycenter, cfwhm, cx)::Profile)(::Type{T2}; normalize = true) where {T2} =
    get_profile(normalize ? Val(:normalize) : Val(:raw), T2, bbox, ycenter, cfwhm, cx)

((; type, bbox, ycenter, cfwhm, cx)::Profile)(bbox2::BoundingBox{Int}; normalize = true) =
    get_profile(normalize ? Val(:normalize) : Val(:raw), type, bbox2, ycenter, cfwhm, cx)


"""
    get_profile(::Type{T}, bbox::BoundingBox, ycenter::Float64, cfwhm::Array, cx::Vector) where {T,N}

Generate a 2D Gaussian-like profile model over the specified bounding box.

The profile models a spectrum as a 1D Gaussianwith position-dependent center and width:
- Center position: `xcenter(y) = Σ cx[i] * ((y - ycenter)^(i-1))`
- FWHM: `width(y) = Σ cfwhm[i,:] * ((y - ycenter)^(i-1))`

For N=2, supports asymmetric profiles with different left/right widths.

# Arguments
- `T`: Output array element type
- `bbox`: Spatial region to evaluate the profile
- `ycenter`: Reference position along dispersion axis
- `cfwhm`: FWHM polynomial coefficients (size: order × N)
- `cx`: Center position polynomial coefficients

# Returns
- `Array{T,2}`: 2D profile image 
"""
function get_profile(
        ::Val{S},
        ::Type{T},
        bbox::BoundingBox{Int64},
        ycenter::Float64,
        cfwhm::Array{Float64, N},
        cx::Vector{Float64}
    ) where {N, T, S}

    xorder = length(cx)
    fwhmorder = size(cfwhm, 1)

    order = max(xorder, fwhmorder)

    ax, ay = axes(bbox)
    ypo = ((ay .- ycenter)) .^ reshape(0:order, 1, order + 1)

    xcenter = view(ypo, :, 1:xorder) * cx

    width = view(ypo, :, 1:fwhmorder) * cfwhm

    fwhm2sigma2 = (1 / (2 * sqrt(2 * log(2))))^2

    if false
        fw = @. T(-1 / (2 * (width^2) * fwhm2sigma2))
        xc = T.(ax .- xcenter')
        if N == 1
            dist = (xc .^ 2) .* reshape(fw, 1, :)
        elseif N == 2
            dist = min.(xc, 0) .^ 2 .* reshape(fw[:, 1], 1, :) .+ max.(xc, 0) .^ 2 .* reshape(fw[:, 2], 1, :)
        else
            error("get_profile : N must be 1 or 2")
        end


        img = exp.(dist)
    else
        img = zeros(T, size(bbox)...)

        @inbounds @simd for j in axes(img, 2) #40
            for i in axes(img, 1) #5
                d = (ax[i] - xcenter[j])
                if N == 1
                    dist = T(-d^2 / (2 * (width[j]^2 * fwhm2sigma2)))
                elseif N == 2
                    dist = T(- d^2 / (2 * (width[j, ifelse(d < 0, 1, 2)]^2 * fwhm2sigma2)))
                else
                    error("get_profile : N must be 1 or 2")
                end
                img[i, j] = exp(dist)
            end
        end
    end
    if S == :normalize
        return img ./ sum(img; dims = 1)
    end
    return img
end

"""
    get_bbox(center_x::Float64, center_y::Float64; bbox_params::BboxParams = BboxParams())

Generate a bounding box centered on the given coordinates.

Creates a rectangular region around the specified center, with dimensions
and offsets determined by the bbox_params configuration.

# Arguments
- `center_x`, `center_y`: Center coordinates
- `bbox_params`: Configuration for bounding box dimensions

# Returns
- `BoundingBox{Int}` if valid, `missing` if out of detector bounds
"""
function get_bbox(center_x::Float64, center_y::Float64; bbox_params::BboxParams = BboxParams())
    @unpack_BboxParams bbox_params
    bbox = round(
        Int,
        BoundingBox(;
            xmin = center_x - BBOX_DX_LOWER,
            xmax = center_x + BBOX_DX_UPPER,
            ymin = center_y - BBOX_DY_LOWER,
            ymax = center_y + BBOX_DY_UPPER
        ),
        RoundNearestTiesUp
    ) # rounding mode to preserve bbox size

    size(bbox) == (BBOX_WIDTH, BBOX_HEIGHT) || return missing
    ((bbox.xmin ≥ 1) & (bbox.xmax ≤ 2048) & (bbox.ymin ≥ 1) & (bbox.ymax ≤ 2048)) || return missing
    return bbox
end


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
    get_wavelength(coefs, ref, pixel)

Convert pixel coordinates to wavelength using polynomial calibration.

Computes: `λ = Σ coefs[i] * ((pixel - ref)/ref)^(i-1)`

# Arguments
- `coefs`: Polynomial coefficients for wavelength solution
- `ref`: Reference pixel position
- `pixel`: Pixel coordinates to convert

# Returns
Wavelength values corresponding to input pixels
"""

function get_wavelength(
        coefs::Vector{<:Union{Nothing, Vector{Float64}}},
        reference_pixel,
        pixel;
        ntasks = 4 * Threads.nthreads()
    )

    wvlngth = tmap(coefs; ntasks = ntasks) do coef
        if isnothing(coef)
            return nothing
        else
            return get_wavelength(coef, reference_pixel, pixel)
        end
    end
    return collect(wvlngth)
end

get_wavelength(coefs, reference_pixel, pixel) =
    get_wavelength(Val(length(coefs) - 1), Val(length(pixel)), coefs, reference_pixel, pixel)

function get_wavelength(::Val{order}, ::Val{len}, coefs, reference_pixel, pixel) where {order, len}
    fullA = SMatrix{len, order + 1}(((pixel .- reference_pixel) ./ reference_pixel) .^ reshape(0:order, 1, :))
    return fullA * coefs
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
        restrict = 0,
        nonnegative::Bool = false,
        ntasks = 4 * Threads.nthreads()
    ) where {T <: Real, N, T2 <: Real, M}

    (1 < N <= 3) || error("extract_spectra: data must have 2 or 3 dimensions")
    valid_lenslets = map(!isnothing, profiles)
    profile_type = ZippedVector{WeightedValue{T2}, 2, true, Tuple{Array{T2, N - 1}, Array{T2, N - 1}}}
    spectra = Vector{Union{profile_type, Nothing}}(undef, length(profiles))
    fill!(spectra, nothing)
    tforeach(findall(valid_lenslets); ntasks = ntasks) do i
        spectra[i] = extract_spectrum(data, profiles[i]; restrict = restrict, nonnegative = nonnegative)
    end
    return spectra
end


"""
    filter_spectra_outliers!(spectra; threshold=3)

Remove outliers from extracted spectra using robust statistics.

Identifies and zeros out spectral points that deviate more than `threshold` 
median absolute deviations from the median spectrum across all traces.

# Arguments
- `spectra`: Vector of WeightedArray spectra (modified in-place)
- `threshold`: Outlier detection threshold in MAD units

Sets precision to 0 and value to 0 for detected outliers.
"""
function filter_spectra_outliers!(
        spectra;
        threshold = 3
    )
    valid_spectra = findall(!isnothing, spectra)
    q = hcat([spectra[i].value for i in valid_spectra ]...)
    m = median(q; dims = 2) # median spectrum
    s = mad.(eachslice(q, dims = 1))
    bad = @. !((m - threshold * s) <= q <= (m + threshold * s))
    for i in eachindex(IndexCartesian(), bad)
        if bad[i]
            spectra[valid_spectra[i[2]]].precision[i[1]] = 0
            spectra[valid_spectra[i[2]]].value[i[1]] = 0
        end
    end

    return
end
