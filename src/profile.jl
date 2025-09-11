struct Profile{N}
    bbox::BoundingBox{Int64}
    ycenter::Float64
    cfwhm::Array{Float64, N}
    cx::Vector{Float64}
end

Profile(bbox::BoundingBox{Int}, cfwhm::AbstractArray, cx::AbstractVector) =
    Profile(bbox, mean(axes(bbox, 2)), cfwhm, cx)

((; bbox, ycenter, cfwhm, cx)::Profile)() = get_profile(bbox, ycenter, cfwhm, cx)


((; bbox, ycenter, cfwhm, cx)::Profile)(bbox2::BoundingBox{Int}) = get_profile(bbox2, ycenter, cfwhm, cx)

function get_profile(bbox::BoundingBox{Int64}, ycenter::Float64, cfwhm::Array{Float64, N}, cx::Vector{Float64}) where {N}


    xorder = length(cx)
    fwhmorder = size(cfwhm, 1)

    order = max(xorder, fwhmorder)

    ax, ay = axes(bbox)
    ypo = ((ay .- ycenter)) .^ reshape(0:order, 1, order + 1)

    xcenter = ypo[:, 1:xorder] * cx

    width = ypo[:, 1:fwhmorder] * cfwhm

    fwhm2sigma = 1 / (2 * sqrt(2 * log(2)))
    fw = @. -1 / (2 * (width * fwhm2sigma)^2)

    xc = (ax .- xcenter')
    if N == 1
        xc = (ax .- xcenter')
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


function extract_model(
        data::WeightedArray{T, N},
        profile::Profile;
        restrict = 0.01,
        nonnegative = false,
        relative = false
    ) where {T, N}
    bbox = profile.bbox
    if relative
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
        model .*= (model .> restrict)
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
