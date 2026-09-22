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


function build_crosstalk_model(
        profiles::AbstractVector{<:Profile{T}},
        template,
        λ,
        transmission = nothing;
        ntasks = 4 * Threads.nthreads()
    ) where {T}

    detector = BoundingBox(1:2048, 1:2048)
    Xtalk_model = zeros(T, size(detector))

    models = build_spectral_models(profiles, template, λ, transmission; ntasks = ntasks)

    for (idx, profile) in enumerate(profiles)
        if !is_profile(profile)
            continue
        end
        bbox = profile.bbox
        bboxL = TwoDimensional.grow(bbox, 6, 0) ∩ detector

        A = OffsetArray(zeros(T, size(bboxL)), axes(bboxL)...)
        OffsetArrays.no_offset_view(view(A, bboxL)) .= profile(bboxL) .* reshape(models[idx], 1, :)
        view(A, bbox) .= zero(T)
        view(Xtalk_model, bboxL) .+= OffsetArrays.no_offset_view(A)
    end
    return Xtalk_model
end

function diagAtA_estimation(A::LinOp, K::Int = 10)
    # Estimate the diagonal of A'A using Girard's method
    spA = LinOps.inputspace(A)
    diag_estimate = zeros(spA)
    for _ in 1:K
        z = rand((-1, 1), spA)
        diag_estimate .+= (A' * A * z) .* z
    end
    return diag_estimate ./ K
end


function build_spectral_models(
        profiles::AbstractVector{<:Union{Profile, LensletError}},
        template,
        λ,
        transmission = nothing;
        ntasks = 4 * Threads.nthreads()
    )

    models = Vector{Union{Nothing, Vector{Float64}}}(undef, length(profiles))
    fill!(models, nothing)
    profile_wavelength = get_wavelength(profiles)
    if transmission === nothing
        tforeach(findall(is_profile, profiles); ntasks = ntasks) do i
            MI = build_sparse_interpolation_integration_matrix(λ, get_lower_uppersamples(profile_wavelength[i])...)
            models[i] = (MI * template)
        end
    else
        tforeach(findall(is_profile, profiles); ntasks = ntasks) do i
            MI = build_sparse_interpolation_integration_matrix(λ, get_lower_uppersamples(profile_wavelength[i])...)
            models[i] = (MI * template) .* transmission[i]
        end
    end

    return models
end
