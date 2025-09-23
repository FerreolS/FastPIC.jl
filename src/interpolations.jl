"""
    get_lower_uppersamples(λ::AbstractVector)

Compute lower and upper bounds for wavelength bins from sample points.

Creates bin edges by taking midpoints between adjacent samples, with extrapolated
edges at the boundaries using linear extrapolation.

# Arguments
- `λ::AbstractVector`: Wavelength sample points (assumed to be monotonically increasing)

# Returns
- `Tuple{Vector, Vector}`: Lower and upper bin edges for each sample point

# Algorithm
- Interior edges: midpoints between adjacent samples `(λ[i] + λ[i+1])/2`
- Boundary edges: linear extrapolation `(3λ[1] - λ[2])/2` and `(3λ[end] - λ[end-1])/2`

# Examples
```julia
λ = [1.0, 2.0, 3.0, 4.0]
lower, upper = get_lower_uppersamples(λ)
# lower = [0.5, 1.5, 2.5, 3.5]
# upper = [1.5, 2.5, 3.5, 4.5]
```
"""
function get_lower_uppersamples(λ::AbstractVector)
    lower = [(3 * λ[1] .- λ[2]) / 2; (λ[1:(end - 1)] .+ λ[2:end]) / 2]
    upper = [(λ[2:end] .+ λ[1:(end - 1)]) / 2; (3 * λ[end] .- λ[end - 1]) / 2]
    return lower, upper
end

"""
    reverse_cumsum(v)

Compute reverse cumulative sum of a vector.

Equivalent to `reverse(cumsum(reverse(v)))` but more efficient.
The result satisfies: `out[i] = sum(v[i:end])`.

# Arguments
- `v`: Input vector

# Returns
- Vector of same type and size as input with reverse cumulative sums

# Examples
```julia
reverse_cumsum([1, 2, 3, 4])  # Returns [10, 9, 7, 4]
```
"""
function reverse_cumsum(v)
    out = similar(v)
    out[end] = v[end]
    @inbounds for i in (length(v) - 1):-1:1
        out[i] = out[i + 1] + v[i]
    end
    return out
end

"""
    build_sparse_interpolation_integration_matrix(knots, lowersample, uppersamples; kernel::Kernel{T,N} = CatmullRomSpline()) where {T,N}

Build a sparse matrix for integrating interpolated functions over bins.

Constructs a matrix `A` such that `A * f` gives the integral of the interpolated
function `f` (defined at `knots`) over each bin defined by `[lowersample[i], uppersamples[i]]`.

# Arguments
- `knots`: Grid points where the function values are defined
- `lowersample`: Lower bounds of integration bins
- `uppersamples`: Upper bounds of integration bins  
- `kernel`: Interpolation kernel (default: Catmull-Rom spline)

# Returns
- `SparseMatrixCSC`: Integration matrix of size `(length(lowersample), length(knots))`

# Algorithm
Uses the interpolation kernel to compute exact integrals of the interpolated function
over each bin by integrating the kernel weights between the bin boundaries.

# Examples
```julia
knots = 1.0:0.1:10.0
lower = [1.05, 2.15, 3.25]
upper = [1.95, 2.85, 3.75]
A = build_sparse_interpolation_integration_matrix(knots, lower, upper)
# A * f gives integrals of interpolated f over the three bins
```
"""
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


"""
    find_index(knots::AbstractRange, sample)

Find the fractional index of a sample point within a uniform grid.

For a uniform grid defined by `knots`, computes the continuous index position
where `sample` would be located. Returns 1.0 for the first knot, 2.0 for the
second knot, etc., with fractional values for intermediate positions.

# Arguments
- `knots::AbstractRange`: Uniform grid points
- `sample`: Point to locate within the grid

# Returns
- `Float64`: Fractional index position (1-based)

# Examples
```julia
knots = 1.0:0.5:5.0  # [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
find_index(knots, 2.25)  # Returns 3.5 (halfway between indices 3 and 4)
```

# Notes
This function assumes `knots` is a uniform range. For non-uniform grids,
use a different search algorithm.
"""
function find_index(knots::AbstractRange, sample)
    return (sample - first(knots)) / step(knots) + 1
end
