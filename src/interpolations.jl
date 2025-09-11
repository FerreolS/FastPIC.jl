function get_lower_uppersamples(λ::AbstractVector)
    lower = [(3 * λ[1] .- λ[2]) / 2; (λ[1:(end - 1)] .+ λ[2:end]) / 2]
    upper = [(λ[2:end] .+ λ[1:(end - 1)]) / 2; (3 * λ[end] .- λ[end - 1]) / 2]
    return lower, upper
end


function reverse_cumsum(v)
    out = similar(v)
    out[end] = v[end]
    @inbounds for i in (length(v) - 1):-1:1
        out[i] = out[i + 1] + v[i]
    end
    return out
end

function build_sparse_interpolation_integration_matrix(knots, lowersample, uppersamples; kernel::Kernel{T, N} = CatmullRomSpline()) where {T, N}

    lk = length(kernel)
    lin = length(uppersamples)
    lin == length(lowersample) || throw(DimensionMismatch("uppersamples and lowersample must have the same length"))
    col = length(knots)

    nelement = col * lin
    L = zeros(Int, nelement)
    C = zeros(Int, nelement)
    V = zeros(T, nelement)
    c = 1

    for (l, (lsample, usample)) in enumerate(zip(lowersample, uppersamples))
        uoffweights = InterpolationKernels.compute_offset_and_weights(kernel, T.(find_index(knots, usample)))
        loffweights = InterpolationKernels.compute_offset_and_weights(kernel, T.(find_index(knots, lsample)))
        uweights = vcat(uoffweights[2]...)[2:end]
        uoff::Int = round(Int, uoffweights[1]) + 1

        lweights = vcat(loffweights[2]...)[2:end]
        loff::Int = round(Int, loffweights[1]) + 1

        lv = uoff - loff + lk - 1
        v = ones(T, lv)
        v[(lv - lk + 2):end] .= reverse_cumsum(uweights)
        v[1:(lk - 1)] .-= reverse_cumsum(lweights)
        off = min.(max.((loff + 1):(loff + lv), 1), col)
        L[c:(c + lv - 1)] .= l
        C[c:(c + lv - 1)] .= off
        V[c:(c + lv - 1)] .= v
        c += lv
    end
    return sparse(L[1:(c - 1)], C[1:(c - 1)], V[1:(c - 1)], lin, col)
end


function find_index(knots::AbstractRange, sample)
    return (sample - first(knots)) / step(knots) + 1
end
