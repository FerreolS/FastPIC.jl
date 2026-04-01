function get_lensletmap(profiles)
    lensletmap = zeros(Int, 2048, 2048)
    for (i, p) in enumerate(profiles)
        if isnothing(p)
            continue
        end
        lensletmap[p.bbox] .= i
    end
    return lensletmap
end

function transform_grid(grid, offset, scale, θ)
    R = @SArray [
        cos(θ) -sin(θ)
        sin(θ) cos(θ)
    ]
    return scale .* (R * grid) .+ offset
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

    cost(x) = h(g(x))
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

function build_grid(halflen; lensletmap = nothing, scale = 15.0, θ = 0.0, center = [1024.0, 1024.0])
    B = @SArray [
        1 1 / 2
        0 √3 / 2
    ]

    R = @SArray [
        cos(θ) -sin(θ)
        sin(θ) cos(θ)
    ]
    idx = -halflen:halflen
    grid = Array{Float64}(undef, 2, length(idx)^2)
    n = 1
    for v in Iterators.product(idx, idx)
        coord = B * vcat(v...)
        dcoord = scale .* (R * coord) .+ center
        1 .≤ (dcoord[1]) < 2048 || continue
        1 .≤ (dcoord[2]) < 2048 || continue

        if !isnothing(lensletmap)
            idx = round.(Int, dcoord)
            if lensletmap[idx...] == 0
                continue
            end
        end
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

function find_lenslet_position!(profiles; laser_models = nothing, halflensequence = (1, 5, 15, 25, 150), maxeval = 1000, verbose = 0, scale = 15.0, θ = 0.0, offset = nothing, center = [1024.0, 1024.0])
    valid = findall(!isnothing, profiles)
    lensletmap = get_lensletmap(profiles)
    if isnothing(laser_models)
        centers = build_centers(profiles, valid)
    else
        centers = build_centers(profiles, laser_models, valid)
    end
    kdtree = KDTree(centers)

    centeridx, _ = knn(kdtree, center, 1)
    if isnothing(offset)
        offset = centers[:, centeridx[:]][:]
    end
    x = vcat(offset..., scale..., θ)

    for hl in halflensequence
        grid = build_grid(hl; lensletmap = lensletmap, center = x[1:2], scale = x[3], θ = x[4])
        adjust_grid_param!(kdtree, grid, x; maxeval = maxeval, verbose = (verbose > 1))
    end
    (verbose > 0) && @info "Final grid parameters: offset=($(x[1]), $(x[2])), scale=$(x[3]), θ=$(rem2pi(x[4], RoundNearest)))"
    grid = transform_grid(build_grid(150; center = x[1:2], scale = x[3], θ = x[4]), x)
    gridtree = KDTree(grid)
    idx, _ = knn(gridtree, centers, 1)
    positions = grid[:, vcat(idx...)]
    for (n, i) in enumerate(valid)
        Accessors.@reset profiles[i].position .= tuple(positions[:, n]...)
    end
    return profiles, x[3]
end #= function find_lenslet_position! =#
