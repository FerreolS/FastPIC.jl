"""
    Profile{T,N,C}

A parametric model of the spectrum of each SPHERE/IFS lenslet.

# Fields
- `type::Type{T}`: Numeric type for computations (e.g., Float64)
- `bbox::BoundingBox{Int64}`: Bounding box defining the spatial extent on the lenslet on the detector
- `ycenter::Float64`: Central position along the dispersion axis
- `cfwhm::Array{Float64,N}`: Polynomial coefficients for FWHM variation (N=1 symmetric, N=2 asymmetric)
- `cx::Vector{Float64}`: Polynomial coefficients for lateral center position variation
- `spectral_coefs::C`: Polynomial coefficients for wavelength solution (can be `nothing` if not calibrated)

# Examples
```julia
bbox = BoundingBox(xmin=10, xmax=50, ymin=100, ymax=150)
cfwhm = [2.5, 0.1]  # FWHM coefficients
cx = [25.0, 0.05]   # Center position coefficients
profile = Profile(Float64, bbox, 125.0, cfwhm, cx)
```
"""
struct Profile{T, N, C}
    type::Type{T}
    bbox::BoundingBox{Int64}
    ycenter::Float64
    cfwhm::Array{Float64, N}
    cx::Vector{Float64}
    spectral_coefs::C
    position::Tuple{Float64, Float64}
    function Profile(T::Type, bbox::BoundingBox{Int}, ycenter::Real, cfwhm::AbstractArray{T2, N}, cx::AbstractVector, spectral_coefs::C, position::NTuple{2, Float64} = (0.0, 0.0)) where {N, T2 <: Real, C <: Union{Nothing, <:AbstractVector{Float64}}}
        @assert T <: Real
        size(cfwhm, 1) ≥ 1 || throw(ArgumentError("cfwhm must have at least one row"))
        length(cx) ≥ 1 || throw(ArgumentError("cx must have at least one element"))
        return new{T, N, C}(T, bbox, ycenter, collect(cfwhm), collect(cx), spectral_coefs, position)
    end
end

Optimisers.trainable(x::Profile) = (; cfwhm = x.cfwhm, cx = x.cx)
"""
    is_profile(value)

Return `true` when `value` is a `Profile` and `false` for a `LensletError`.
This predicate is intended for filtering lenslet-indexed heterogeneous vectors.
"""
is_profile(value) = value isa Profile

"""
    is_calibrated(profile::Profile)

Return `true` when `profile` has wavelength-calibration coefficients and can be
used by calibrated-only operations such as PIC construction.
"""
is_calibrated(profile::Profile) = profile.spectral_coefs isa AbstractVector{Float64}

function filter_profiles(x::AbstractVector{<:Union{Profile, LensletError}})
    P = Base.typesplit(eltype(x), LensletError)
    good_profile = findall(is_profile, x)
    filtered = P[x[i] for i in good_profile]
    return good_profile, filtered
end

Profile(T::Type, bbox::BoundingBox{Int}, cfwhm::AbstractArray, cx::AbstractVector) =
    Profile(T, bbox, mean(axes(bbox, 2)), cfwhm, cx, nothing)
Profile(T::Type, bbox::BoundingBox{Int}, cfwhm::AbstractArray, cx::AbstractVector, position::NTuple{2}) =
    Profile(T, bbox, mean(axes(bbox, 2)), cfwhm, cx, nothing, position)

#Profile(bbox, ycenter, cfwhm, cx) = Profile(Float64, bbox, ycenter, cfwhm, cx, nothing)

((; type, bbox, ycenter, cfwhm, cx)::Profile)() =
    get_footprint(type, bbox, ycenter, cfwhm, cx)

((; type, bbox, ycenter, cfwhm, cx)::Profile)(::Type{T2}) where {T2} =
    get_footprint(T2, bbox, ycenter, cfwhm, cx)

((; type, ycenter, cfwhm, cx)::Profile)(bbox::BoundingBox{Int}) =
    get_footprint(type, bbox, ycenter, cfwhm, cx)

"""
    get_footprint(::Type{T}, bbox::BoundingBox, ycenter::Float64, cfwhm::Array, cx::Vector) where {T,N}

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
function get_footprint(
        ::Type{T},
        bbox::BoundingBox{Int64},
        ycenter::Float64,
        cfwhm::Array{Float64, N},
        cx::Vector{Float64}
    ) where {T, N}

    N == 1 || error("get_footprint : N ≠ 1 not yet implemented for get_footprint")


    xorder = length(cx)
    fwhmorder = size(cfwhm, 1)

    order = max(xorder, fwhmorder)

    _, ay = axes(bbox)
    ypo = ((ay .- ycenter)) .^ reshape(0:order, 1, order + 1)

    xcenter = view(ypo, :, 1:xorder) * cx

    width = view(ypo, :, 1:fwhmorder) * cfwhm


    img = zeros(T, size(bbox)...)

    ax = (bbox.xmin - 0.5):(bbox.xmax + 0.5)

    @inbounds @simd for j in axes(img, 2)
        integ = Vector{T}(undef, length(ax))
        for i in axes(ax, 1) #5
            integ[i] = T(1 / 2 * erf((ax[i] - xcenter[j]) * 2 * sqrt(log(2)) / width[j]))
        end
        img[:, j] .= diff(integ)
    end

    return img
end

get_footprint((; type, bbox, ycenter, cfwhm, cx)::Profile) =
    get_footprint(type, bbox, ycenter, cfwhm, cx)

get_footprint((; type, ycenter, cfwhm, cx)::Profile, bbox) =
    get_footprint(type, bbox, ycenter, cfwhm, cx)


"""
    get_bbox(center_x::Float64, center_y::Float64; bbox_params::BboxParams = BboxParams())

Generate a bounding box centered on the given coordinates.

Creates a rectangular region around the specified center, with dimensions
and offsets determined by the bbox_params configuration.

# Arguments
- `center_x`, `center_y`: Center coordinates
- `bbox_params`: Configuration for bounding box dimensions

# Returns
- `BoundingBox{Int}` if valid, or `lenslet_out_of_bounds` if outside the detector
"""
function get_bbox(center_x::Real, center_y::Real; bbox_params::BboxParams = BboxParams())
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

    size(bbox) == (BBOX_WIDTH, BBOX_HEIGHT) || return lenslet_out_of_bounds
    ((bbox.xmin ≥ 1) & (bbox.xmax ≤ 2048) & (bbox.ymin ≥ 1) & (bbox.ymax ≤ 2048)) || return lenslet_out_of_bounds
    return bbox
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


function get_wavelength(profile::Profile{T, N, <:AbstractVector{Float64}}) where {T, N}
    return get_wavelength(profile.spectral_coefs, profile.ycenter - profile.bbox.ymin, 1:size(profile.bbox, 2))
end

function get_wavelength(profile::Profile{T, N, Nothing}) where {T, N}
    throw(ArgumentError("profile has not been wavelength-calibrated"))
end

function get_wavelength(profiles::AbstractVector{<:Union{Profile, LensletError}}; ntasks = 4 * Threads.nthreads())
    wvlngth = tmap(profiles; ntasks = ntasks) do p
        if !is_profile(p)
            return p isa LensletError ? p : lenslet_invalid_data
        else
            return get_wavelength(p)
        end
    end
    return collect(wvlngth)
end
