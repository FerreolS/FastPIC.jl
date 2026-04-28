function get_lensletmap(profiles)
    lensletmap = zeros(Int, 2048, 2048)
    for (i, p) in enumerate(profiles)
        if isnothing(p)
            continue
        end
        mask = view(lensletmap, p.bbox) .> 0
        view(lensletmap, p.bbox) .= i
        view(lensletmap, p.bbox)[mask] .= 100000
    end
    return lensletmap
end


"""
    get_meanx(data::WeightedArray{T,N}, bbox; relative=false) where {T,N}

Compute the precision-weighted mean position along the first axis.

Calculates the centroid position using: `Σ(x * √w * I) / Σ(√w * I)`
where x is position, w is precision (inverse variance), and I is intensity.

# Arguments
- `data::WeightedArray{T,N}`: Input weighted data
- `bbox`: Bounding box defining the region of interest
- `relative::Bool = false`

# Returns
- `Float64`: Precision-weighted mean position along the first axis

# Examples
```julia
center_x = get_meanx(detector_data, profile_bbox)
```
"""
function get_meanx(data::WeightedArray{T, N}, bbox; relative = false) where {T, N}
    (; value, precision) = view(data, bbox)

    if relative
        ax = axes(value, 1)
    else
        ax = axes(bbox, 1)
    end

    return sum(value .* sqrt.(precision) .* ax) ./ sum(sqrt.(precision) .* value)
end

function get_maxx(data::WeightedArray{T, N}, bbox; relative = false) where {T, N}
    (; value, precision) = view(data, bbox)

    if relative
        ax = axes(value, 1)
    else
        ax = axes(bbox, 1)
    end

    return ax[argmax(sum(value .* precision, dims = 2) ./ sum(precision, dims = 2))[1]]
end


function transform_grid(grid, offset, scale, θ)
    R = @SArray [
        cos(θ) -sin(θ)
        sin(θ) cos(θ)
    ]
    return scale .* (R * grid) .+ offset
end

function poly_coeffs_size(order::Integer)
    order >= 0 || error("order must be non-negative")
    return (order + 1) * (order + 2) ÷ 2
end

function poly_order_from_coeffs_size(ncoeff::Integer)
    ncoeff >= 0 || error("number of coefficients must be non-negative")
    order_value = (-3 + sqrt(1 + 8 * ncoeff)) / 2
    isinteger(order_value) || error("invalid number of coefficients: $ncoeff")
    order = Int(order_value)
    return order
end

"""
    build_poly_matrix(grid, order) -> A

Build the design matrix for a 2D polynomial fit of degree `order` from a `2×N` grid.

Each column of `A` corresponds to a monomial basis term `x^(deg-p) * y^p`, ordered as:
- column 1: 1  (constant)
- deg=1, p=0..1:  x, y
- deg=2, p=0..2:  x², xy, y²
- deg=3, p=0..3:  x³, x²y, xy², y³
- ...

The column ordering matches `transform_grid_poly`, so coefficients obtained by fitting with
`A` can be used directly with `transform_grid_poly`.

# Returns
- `Matrix{T}` of size `(N, poly_coeffs_size(order))`
"""
function build_poly_matrix(grid::AbstractMatrix{T}, order::Integer) where {T}
    order >= 0 || error("order must be non-negative")
    gx = grid[1, :]
    gy = grid[2, :]
    n = size(grid, 2)
    ncols = poly_coeffs_size(order)
    A = ones(T, n, ncols)

    xpows = Vector{Vector{T}}(undef, order + 1)
    ypows = Vector{Vector{T}}(undef, order + 1)
    xpows[1] = ones(T, n)
    ypows[1] = ones(T, n)
    @inbounds for p in 2:(order + 1)
        xpows[p] = xpows[p - 1] .* gx
        ypows[p] = ypows[p - 1] .* gy
    end

    k = 2
    @inbounds for deg in 1:order
        for p in 0:deg
            A[:, k] .= xpows[deg - p + 1] .* ypows[p + 1]
            k += 1
        end
    end
    return A
end

function filter_grid(grid)
    mask = 1 .< grid[1, :] .< 2048 .&& 1 .< grid[2, :] .< 2048
    grid = grid[:, mask]
    return grid
end

function build_boxes(grid, lamp; threshold = 4, bboxparams = BboxParams(), xshift_bbox = false)
    medlamp = median(lamp.value)
    mask = trues(size(grid, 2))
    boxes = Vector{BoundingBox{Int}}(undef, size(grid, 2))
    @inbounds   for i in axes(grid, 2)
        box = get_bbox(grid[1, i], grid[2, i]; bbox_params = bboxparams)
        if ismissing(box)
            mask[i] = false
            continue
        end
        if mean(view(lamp, box)).value < medlamp / threshold
            mask[i] = false
            continue
        end
        if xshift_bbox
            boxes[i] = get_bbox(get_meanx(lamp, box), grid[2, i]; bbox_params = bboxparams)
        else
            boxes[i] = box
        end
    end
    return grid[:, mask], boxes[mask]
end

function transform_grid(grid, x)
    length(x) == 4 &&  return transform_grid(grid, x[1:2], x[3], x[4])
    length(x) == 5 &&  return transform_grid(grid, x[1:2], x[3:4], x[5])
    error("x should have length 4 or 5")
end

function dist_grid(kdtree::KDTree, grid; maxdist = 14.0)
    _, dists = knn(kdtree, grid, 1)
    return mapreduce(x -> ifelse(x[1] < maxdist, abs(x[1]), maxdist), +, dists)
end

function adjust_grid_param!(
        kdtree::KDTree, grid0, x;
        maxeval = 10_000,
        verbose = false
    )
    g = Base.Fix1(transform_grid, grid0)
    h = Base.Fix1(dist_grid, kdtree)

    cost(x) = h(filter_grid(g(x)))
    Newuoa.optimize!(cost, x, 1.0, 1.0e-9; check = false, maxeval = maxeval, verbose = verbose)
    return x
end

function init_grid_param(kdtree::KDTree, centers; scale = 15.0, θ = 0.0, offset = nothing, center = [1024.0, 1024.0])
    centeridx, _ = knn(kdtree, center, 1)
    if isnothing(offset)
        offset = centers[:, centeridx[:]][:]
    end
    x = vcat(offset..., scale..., θ)
    return x
end

function build_grid(halflen)
    B = @SArray [
        1 1 / 2
        0 √3 / 2
    ]

    idx = -halflen:halflen
    grid = Array{Float64}(undef, 2, length(idx)^2)
    n = 1
    for v in Iterators.product(idx, idx)
        coord = B * vcat(v...)
        grid[:, n] = coord
        n = n + 1
    end
    return grid[:, 1:(n - 1)]
end

function build_centers(profiles, laser_models, valid)
    nb_valid = length(valid)
    centers = zeros(Float64, 2, nb_valid)
    for (n, i) in enumerate(valid)
        centers[1, n] = profiles[i].cx[1]
        centers[2, n] = profiles[i].bbox.ymin + laser_models[i].position[2]
        # or  centers[2, n] = profiles[i].ycenter
    end
    return centers
end
function build_centers(profiles, valid)
    nb_valid = length(valid)
    centers = zeros(Float64, 2, nb_valid)
    for (n, i) in enumerate(valid)
        centers[1, n] = profiles[i].cx[1]
        centers[2, n] = profiles[i].ycenter
    end
    return centers
end

function find_lenslet_position!(profiles; laser_models = nothing, halflensequence = (1, 5, 15, 25, 50, 100, 150, 200), maxeval = 1000, verbose = 0, scale = 15.0, θ = 0.0, offset = nothing, center = [1024.0, 1024.0])
    valid = findall(!isnothing, profiles)
    if isnothing(laser_models)
        centers = build_centers(profiles, valid)
    else
        centers = build_centers(profiles, laser_models, valid)
    end
    grid, x = build_lenslet_grid(centers; halflensequence = halflensequence, maxeval = maxeval, verbose = verbose, scale = scale, θ = θ, offset = offset, center = center)
    gridtree = KDTree(grid)
    idx, _ = knn(gridtree, centers, 1)
    positions = grid[:, vcat(idx...)]
    for (n, i) in enumerate(valid)
        Accessors.@reset profiles[i].position .= tuple(positions[:, n]...)
    end
    return profiles, x[3]
end


function build_lenslet_grid(centers; halflensequence = (1, 5, 15, 25, 50, 100, 150), maxeval = 1000, verbose = 0, scale = 15.0, θ = 0.0, offset = nothing, center = [1024.0, 1024.0])
    kdtree = KDTree(centers)

    centeridx, _ = knn(kdtree, center, 1)
    if isnothing(offset)
        offset = centers[:, centeridx[1]][:]
    end
    x = vcat(offset..., scale..., θ)

    for hl in halflensequence
        grid = build_grid(hl)
        adjust_grid_param!(kdtree, grid, x; maxeval = maxeval, verbose = (verbose > 1))
    end
    (verbose > 0) && @info "Final grid parameters: offset=($(x[1]), $(x[2])), scale=$(x[3]), θ=$(rem2pi(x[4], RoundNearest)))"
    grid = filter_grid(transform_grid(build_grid(150), x))
    return grid, x
end


function estimate_lenslet_warping(grid, centers; verbose = 0, order = 3)
    kdtree = KDTree(grid)
    idx, dist = knn(kdtree, centers, 1)
    xc = [x[1] for x in idx]
    selected = [d[1] .< 14.0 for d in dist]
    xc = xc[selected]
    grd = view(grid, :, xc)
    A = build_poly_matrix(grd, order)
    ML = inv(A' * A) * A'
    px = ML * (grd[1, :] .- centers[1, selected])
    py = ML * (grd[2, :] .- centers[2, selected])
    if verbose > 0
        @info "Estimated lenslet warping polynomial coefficients (order $order):"
        for i in 1:length(px)
            @info "  x coeff $i: $(px[i]), y coeff $i: $(py[i])"
        end
    end
    coefs = hcat(px, py)
    return coefs
end

function correct_lenslet_warping(grid, coefs)
    A = build_poly_matrix(grid, poly_order_from_coeffs_size(size(coefs, 1)))
    correction = A * coefs
    corrected_grid = grid .- correction'
    return corrected_grid
end

function cross_correlation(λ, lamp_spectrum, lasers, lamp, laser_line_width, lasers_λs, lamp_fwhms)
    T = Float64
    F = LinOpDFT(T, (2048, 2048))
    Δλ = (λ[end] - λ[1]) / length(λ)


    lamp_template = gaussian.(lamp_fwhms, T.(-2:2)) .* lamp_spectrum'
    lamp_template ./= sum(lamp_template)
    lamp_template ./= sqrt(sum(lamp_template .^ 2))
    image_template = zeros(T, 2048, 2048)
    image_template[1023:1027, 1004:1047] .= lamp_template[end:-1:1, end:-1:1]
    image_template = fftshift(image_template)

    A = (inv(F) * (F * image_template .* (F * (lamp.value .* lamp.precision))))
    b = (inv(F) * (F * (image_template .^ 2) .* (F * lamp.precision)))

    corrlamp = A ./ b

    fvalue = F * (lasers.value .* lasers.precision)
    fprecision = F * lasers.precision
    for laser_λ in lasers_λs
        laser_template = gaussian.(laser_line_width, T.(-2:2)) .* (gaussian.(laser_line_width * Δλ, T.(λ .- laser_λ)))'
        laser_template ./= sum(laser_template)

        laser_template ./= sqrt(sum(laser_template .^ 2))
        fill!(image_template, zero(T))
        image_template[1023:1027, 1004:1047] .= laser_template[end:-1:1, end:-1:1]
        image_template = fftshift(image_template)
        A .+= (inv(F) * ((F * image_template) .* fvalue))
        b .+= (inv(F) * ((F * (image_template .^ 2)) .* fprecision))
    end
    corr = A ./ b
    map!(x -> ifelse(isfinite(x), x, 0.0), corr)
    map!(x -> ifelse(isfinite(x), x, 0.0), corrlamp)
    return corr, corrlamp
end


function initialize_bboxes(
        lamp, lasers;
        calib_params::FastPICParams = FastPICParams()
    )
    @unpack_FastPICParams calib_params
    @unpack_BboxParams bbox_params

    corr, corrlamp = cross_correlation(λ_template, lamp_template, lasers, lamp, laser_line_width[1], lasers_λs, lamp_cfwhms_init[1])
    medlamp = median(lamp.value)

    centers = filter_grid(transform_grid(build_grid(150), lenslets_offset, lenslets_scale, lenslets_θ))

    valid = falses(size(centers, 2))
    @inbounds for i in axes(centers, 2)
        bbox = get_bbox(centers[1, i], centers[2, i]; bbox_params = bbox_params)
        if ismissing(bbox)
            continue
        end
        if get_value(mean(view(lamp, bbox))) < medlamp / lenslets_threshold
            continue
        end
        valid[i] = true
        idx = argmax(view(corr, bbox))
        idx += CartesianIndex(bbox.xmin - 1, bbox.ymin - 1)
        centers[1, i] = idx[1]
        centers[2, i] = idx[2]
    end
    centers = centers[:, valid]
    grid, x = build_lenslet_grid(centers; offset = lenslets_offset, scale = lenslets_scale, θ = lenslets_θ)
    poly_coefs = estimate_lenslet_warping(grid, centers; order = lenslets_warping_order)
    grid = correct_lenslet_warping(grid, poly_coefs)
    grid, bboxes = build_boxes(grid, lamp; threshold = lenslets_threshold, bboxparams = bbox_params)
    return grid, bboxes, x[3], x[4]

end

function build_lenslet_map(profiles::AbstractVector{<:Profile}; Npix = 300, lenslet_width = 15 / 2048 * 300, pad = 0)
    sz = (Npix + 2 * pad, Npix + 2 * pad)
    lensletmap = -1 .* ones(Int, sz...)
    centers = hcat((vcat(p.position...) for p in profiles)...) ./ 2048 .* Npix .+ pad
    kdtree = KDTree(centers)
    for i in CartesianIndices(lensletmap)
        idx, dist = knn(kdtree, vcat(Tuple(i)...), 1)
        if dist[1] < lenslet_width
            lensletmap[i] = idx[1]
        end
    end
    return lensletmap
end
