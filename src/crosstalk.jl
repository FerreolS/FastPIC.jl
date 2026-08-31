function build_crosstalk_operator(profiles::Vector{<:Profile{T}}) where {T}
    Np = length(profiles)
    Nl = length(get_wavelength(profiles[1]))
    sizein = (Np, Nl)
    Ctlk = compute_cross_talk(profiles)
    Xtlk = LinOpSparse(Ctlk, sizein, sizein)
    return Xtlk
end


function get_neighbor_lenslets(lmap, bbox)
    detector = BoundingBox(1:2048, 1:2048)
    lbox = TwoDimensional.grow(bbox, 1, 0) ∩ detector
    return unique(lmap[axes(lbox)...][[1, end], :])
end

function compute_cross_talk(profiles::Vector{<:Profile{T}}) where {T}
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
        if isnothing(profile)
            continue
        end
        bbox = profile.bbox
        (; ymin, ymax) = bbox
        neighbors = get_neighbor_lenslets(lmap, bbox)
        for neighbor in neighbors
            (neighbor == 0 || neighbor == idx) && continue
            neighbor_profile = profiles[neighbor]
            sharedy = (ymin:ymax) ∩ axes(neighbor_profile.bbox, 2)
            nrange = (sharedy.start - ymin + 1):(sharedy.stop - ymin + 1)
            sharedbbx = BoundingBox(axes(bbox, 1), sharedy)
            nbrbbx = BoundingBox(axes(neighbor_profile.bbox, 1), sharedy)
            nbr_range = (sharedy.start - neighbor_profile.bbox.ymin + 1):(sharedy.stop - neighbor_profile.bbox.ymin + 1)
            values = sum((neighbor_profile(sharedbbx; normalize = false) ./ sum(neighbor_profile(nbrbbx; normalize = false); dims = 1)) .* profile(sharedbbx); dims = 1)[:]
            append!(I, (nrange .- 1) .* nprofiles .+ idx)
            append!(J, (nbr_range .- 1) .* nprofiles .+ neighbor)
            append!(V, values)
        end
    end
    return sparse(I, J, V, nelements, nelements) .+ sparse(LinearAlgebra.I, nelements, nelements)
end


struct LinOpSparse{I, O, S <: AbstractMatrix} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    sparse_matrix::S
end

function LinOpSparse(matrix, sizein, sizeout)
    size(matrix, 1) == prod(sizeout) || throw(DimensionMismatch("Matrix row size does not match output size"))
    size(matrix, 2) == prod(sizein) || throw(DimensionMismatch("Matrix column size does not match input size"))
    return LinOpSparse(LinOps.CoordinateSpace(sizein), LinOps.CoordinateSpace(sizeout), matrix)
end

Base.eltype(A::LinOpSparse) = eltype(A.sparse_matrix)


function LinOps.apply_(A::LinOpSparse, x)
    return reshape(A.sparse_matrix * reshape(x, :), size(A.outputspace))
end

function LinOps.apply_!(y, A::LinOpSparse, x)
    ry = reshape(y, :)
    mul!(ry, A.sparse_matrix, reshape(x, :))
    return y
end

function LinOps.apply_adjoint_(A::LinOpSparse, x)
    return reshape(A.sparse_matrix' * reshape(x, :), size(A.inputspace))
end

function LinOps.apply_adjoint_!(y, A::LinOpSparse, x)
    ry = reshape(y, :)
    mul!(ry, A.sparse_matrix', reshape(x, :))
    return y
end
