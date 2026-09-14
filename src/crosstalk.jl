function build_crosstalk_operator(profiles::AbstractVector{<:Profile})
    Np = length(profiles)
    Nl = length(get_wavelength(profiles[1]))
    sizein = (Np, Nl)
    Ctlk = build_crosstalk_matrix(profiles)
    Xtlk = LinOpSparse(Ctlk, sizein, sizein)
    return Xtlk
end


function get_neighbor_lenslets(lmap, bbox)
    detector = BoundingBox(1:2048, 1:2048)
    lbox = TwoDimensional.grow(bbox, 2, 0) ∩ detector
    return unique(view(lmap, lbox)[[1, end], :])
end

get_neighbor_lenslets(lmap, (; bbox)::Profile) = get_neighbor_lenslets(lmap, bbox)

function build_crosstalk_matrix(profiles::AbstractVector{<:Profile{T}}) where {T}
    lmap = get_lensletmap(profiles)
    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{T}()
    nprofiles = length(profiles)
    nelements = nprofiles * length(axes(profiles[1].bbox, 2))
    sizehint!(I, nelements * 2)
    sizehint!(J, nelements * 2)
    sizehint!(V, nelements * 2)
    for (idx, profile) in enumerate(profiles)
        if !is_profile(profile)
            continue
        end
        bbox = profile.bbox
        (; ymin, ymax) = bbox
        neighbors = get_neighbor_lenslets(lmap, bbox)
        for neighbor in neighbors
            (neighbor == 0 || neighbor == idx) && continue
            neighbor_profile = profiles[neighbor]
            sharedy = axes(bbox, 2) ∩ axes(neighbor_profile.bbox, 2)
            nrange = (sharedy.start - ymin + 1):(sharedy.stop - ymin + 1)
            sharedbbx = BoundingBox(axes(bbox, 1), sharedy)
            nbr_range = (sharedy.start - neighbor_profile.bbox.ymin + 1):(sharedy.stop - neighbor_profile.bbox.ymin + 1)
            values = sum(neighbor_profile(sharedbbx; normalize = true) .* profile(sharedbbx); dims = 1)[:]
            append!(I, (nrange .- 1) .* nprofiles .+ idx)
            append!(J, (nbr_range .- 1) .* nprofiles .+ neighbor)
            append!(V, values)
        end
    end
    return sparse(I, J, V, nelements, nelements) .+ sparse(LinearAlgebra.I, nelements, nelements)
end
