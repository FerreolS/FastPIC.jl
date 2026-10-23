function build_neighborhood(lenslet_width, lenslet_θ; radius = 1, width = 0, height = 0, depth = 1)
    neighb(i) = lenslet_width .* reverse(sincos(lenslet_θ .+ i .* (60 / 180 * π)))

    if depth < 0
        neighboring_lenslets = CartesianIndex.(vcat((0, 0), [round.(Int, j .* neighb(i)) for i in 1:6, j in 1:radius]...))
        searchdirection = CartesianIndices((-width:width, -height:height))
        return (neighboring_lenslets .+ reshape(searchdirection, 1, 2width + 1, 2height + 1))[:]
    else
        neighboring_lenslets = CartesianIndex.(vcat((0, 0, 0), [tuple(round.(Int, j .* neighb(i))..., 0) for i in 1:6, j in 1:radius]...))
        searchdirection = CartesianIndices((-width:width, -height:height, -depth:depth))
        return (neighboring_lenslets .+ reshape(searchdirection, 1, 2width + 1, 2height + 1, 2depth + 1))[:]

    end
end

function nonlocalmedian(
        values::AbstractArray{T, M},
        neighborhood::AbstractArray{CartesianIndex{M}},
        selected_area = nothing
    ) where {T, M}
    _median = similar(values)
    _mad = similar(values)
    nonlocalmedian!(_median, _mad, values, neighborhood, selected_area)
    return _median, _mad
end


function nonlocalmedian!(
        _median::AbstractArray{T, M},
        _mad::AbstractArray{T, M},
        values::AbstractArray{T, M},
        neighborhood::AbstractArray{CartesianIndex{M}},
        selected_area = nothing,
    ) where {T, M}
    axes(values) == axes(_median) || throw(DimensionMismatch("values and _median must have the same axes"))
    axes(values) == axes(_mad) || throw(DimensionMismatch("values and _mad must have the same axes"))
    Base.require_one_based_indexing(_median, _mad, values)
    backend = KernelAbstractions.get_backend(values)

    if isnothing(selected_area)
        selected_area = CartesianIndices(ntuple(n -> axes(values, n), M))
    end

    _nlmedian_kernel!(backend)(
        _median,
        _mad,
        values,
        neighborhood,
        selected_area,
        Val(length(neighborhood));
        ndrange = size(values)
    )
    KernelAbstractions.synchronize(backend)

    return _median, _mad
end


@kernel  function _nlmedian_kernel!(
        _median,
        _mad,
        @Const(array::AbstractArray{T, M}),
        neighborhood::NT,
        selected_area::ST,
        ::Val{C}
    ) where {C, T, M, NT, ST}
    index = @index(Global, Cartesian)

    # buf = MVector{C, T}(undef)
    # dev = MVector{C, T}(undef)
    buf = @private T (C,)
    dev = @private T (C,)

    counter = 0

    if NT <: CartesianIndices && ST <: CartesianIndices
        @inbounds for candidate in (neighborhood .+ index) ∩ selected_area
            counter += 1
            buf[counter] = array[candidate]
        end
    else
        @inbounds for k in eachindex(neighborhood)
            candidate = neighborhood[k] + index
            candidate ∈ selected_area || continue
            counter += 1
            @inbounds buf[counter] = array[candidate]
        end
    end

    @inbounds med = _sorted_median!(buf, counter)
    @inbounds for i in 1:counter
        dev[i] = abs(buf[i] - med)
    end
    # same normalisation as StatsBase.mad (normalize = true)
    @inbounds _mad[index] = _sorted_median!(dev, counter) * T(1.482602218505602)
    _median[index] = med
end

# In-place insertion sort of x[1:n] followed by median; allocation free (n is small).
@inline Base.@propagate_inbounds function _sorted_median!(x, n)
    for i in 2:n
        v = x[i]
        j = i - 1
        while j >= 1 && x[j] > v
            x[j + 1] = x[j]
            j -= 1
        end
        x[j + 1] = v
    end
    n == 0 && return oftype(x[1], NaN)
    h = (n + 1) >> 1
    return isodd(n) ? x[h] : (x[h] + x[h + 1]) / 2
end

function refine_bad_pixels!(
        data::WeightedArray{T, M},
        lenslet_width,
        lenslet_θ;
        radius = 1,
        width = 0,
        height = 0,
        depth = size(data, 3),
        threshold = 5
    ) where {T, M}
    neighborhood = build_neighborhood(lenslet_width, lenslet_θ; radius = radius, width = width, height = height, depth = depth)
    median, mad = nonlocalmedian(data.value, neighborhood)
    bad_pixels = abs.(get_value(data) .- median) .> (threshold * mad)
    data.precision[bad_pixels] .= T(0)
    nulprecision = data.precision .== T(0)
    data.value[nulprecision] .= median[nulprecision]
    return data
end
