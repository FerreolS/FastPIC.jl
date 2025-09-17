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


((; type, bbox, ycenter, cfwhm, cx)::Profile)() = get_profile(type, bbox, ycenter, cfwhm, cx)
((; type, bbox, ycenter, cfwhm, cx)::Profile)(::Type{T2}) where {T2} = get_profile(T2, bbox, ycenter, cfwhm, cx)

((; type, bbox, ycenter, cfwhm, cx)::Profile)(bbox2::BoundingBox{Int}) =
    get_profile(type, bbox2, ycenter, cfwhm, cx)

function get_profile(
        ::Type{T},
        bbox::BoundingBox{Int64},
        ycenter::Float64,
        cfwhm::Array{Float64, N},
        cx::Vector{Float64}
    ) where {N, T}

    xorder = length(cx)
    fwhmorder = size(cfwhm, 1)

    order = max(xorder, fwhmorder)

    ax, ay = axes(bbox)
    ypo = ((ay .- ycenter)) .^ reshape(0:order, 1, order + 1)

    xcenter = ypo[:, 1:xorder] * cx

    width = ypo[:, 1:fwhmorder] * cfwhm

    fwhm2sigma = 1 / (2 * sqrt(2 * log(2)))
    fw = @. T(-1 / (2 * (width * fwhm2sigma)^2))

    xc = T.(ax .- xcenter')
    if N == 1
        dist = (xc .^ 2) .* reshape(fw, 1, :)
    elseif N == 2
        dist = min.(xc, 0) .^ 2 .* reshape(fw[:, 1], 1, :) .+ max.(xc, 0) .^ 2 .* reshape(fw[:, 2], 1, :)
    else
        error("get_profile : N must be 1 or 2")
    end


    img = exp.(dist)
    return img #./ sum(img; dims=1)
end


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

get_wavelength(coefs, ref, pixel) = get_wavelength(Val(length(coefs) - 1), Val(length(pixel)), coefs, ref, pixel)

function get_wavelength(::Val{order}, ::Val{len}, coefs, ref, pixel) where {order, len}
    fullA = SMatrix{len, order + 1}(((pixel .- ref) ./ ref) .^ reshape(0:order, 1, :))
    return fullA * coefs
end

function extract_spectra(
        data::WeightedArray{T, N},
        profiles::Vector{Union{Nothing, Profile{T2, <:Int}}};  # allow any M
        restrict = 0,
        nonnegative::Bool = false,
        multi_thread::Bool = true
    ) where {T <: Real, N <: Int, T2 <: Real}
    (1 < N <= 3) || error("extract_spectra: data must have 2 or 3 dimensions")
    valid_lenslets = map(!isnothing, profiles)
    profile_type = ZippedVector{WeightedValue{T2}, 2, true, Tuple{Array{T2, N - 1}, Array{T2, N - 1}}}
    spectra = Vector{Union{profile_type, Nothing}}(undef, length(profiles))
    fill!(spectra, nothing)
    #Threads.@threads for i in findall(valid_lenslets)
    # from https://discourse.julialang.org/t/optionally-multi-threaded-for-loop/81902/8?u=skleinbo
    _foreach = multi_thread ? OhMyThreads.tforeach : Base.foreach
    _foreach(findall(valid_lenslets)) do i
        spectra[i] = extract_spectrum(data, profiles[i]; restrict = restrict, nonnegative = nonnegative)
    end
    return spectra
end
