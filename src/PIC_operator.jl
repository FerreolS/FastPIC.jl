function build_PIC_operators(profiles, Npix, λ, lenslet_width; T = Float64, pad::Int = 0)

    paddedinpix = 2 * pad * 2048 / Npix + 2048

    sz = (Npix + 2 * pad, Npix + 2 * pad, length(λ))
    # cx = [ p.position[1] for p in profiles] ./ paddedinpix .* 2π
    # cy = [ p.position[2] for p in profiles] ./ paddedinpix .* 2π

    cx = ([ p.position[1] for p in profiles] .+ (pad * 2048 / Npix)) ./ paddedinpix .* 2π
    cy = ([ p.position[2] for p in profiles] .+ (pad * 2048 / Npix)) ./ paddedinpix .* 2π
    points = (cx, cy)
    lenslet_radius = lenslet_width / 2048 * Npix / 2

    if T <: Complex
        r2c = false
    else
        r2c = true
    end

    mtf = compute_airy_mtf(Npix + 2 * pad, lenslet_radius; normalize = true, r2c = r2c)
    D = LinOpDiag(mtf)
    sz2 = (round(Int, sz[1] / 2) + 1, sz[2:end]...)
    C = LinOpMapslice(sz2, D, [1, 2]) * LinOpDFT(T, sz, dims = [1, 2])
    NF = LinOpNFFT(T, sz[1:2], points; sort_points = True())
    II = LinOpMapslice(sz2, NF', [1, 2])

    MI = [ FastPIC.build_sparse_interpolation_integration_matrix(get_precision(T), λ, profile) for profile in profiles]

    P = LinOpMapslice(outputsize(II), MI, 2)
    PIC = P * II * C * UniformScaling(get_precision(T)(2 / ((Npix + 2 * pad)^2)))
    return PIC

end

function compute_airy_mtf(len, radius; normalize = true, r2c = false)
    if r2c
        x = rfftfreq(len)
    else
        x = fftfreq(len)
    end
    y = fftfreq(len)
    r = sqrt.(x .^ 2 .+ y' .^ 2)
    mtf = 2 * besselj1.(2π * radius * r) ./ (2π * radius * r)
    mtf[1, 1] = 1.0
    if !normalize
        mtf .*= 2π * radius^2
    end
    return mtf
end

function build_LinOpIntegration_operators(profiles, λ; T = Float64)

    Np = length(profiles)
    Np == 0 && throw(ArgumentError("No profiles provided"))
    Nλ = length(λ)
    Nl = length(get_wavelength(profiles[1]))

    Lc = Vector{Vector{Int}}(undef, Np)
    Cc = Vector{Vector{Int}}(undef, Np)
    Vc = Vector{Vector{T}}(undef, Np)

    for (i, p) in enumerate(profiles)
        Lp, Cp, Vp = build_sparse_interpolation_integration_coordinate_list(get_precision(T), λ, p)
        Lc[i] = Lp
        Cc[i] = Cp
        Vc[i] = Vp
    end
    Nel = maximum(length(l) for l in Lc)
    L = ones(Int, Nel, Np)
    C = ones(Int, Nel, Np)
    V = zeros(T, Nel, Np)
    for i in 1:Np
        len = length(Lc[i])
        L[1:len, i] .= Lc[i]
        C[1:len, i] .= Cc[i]
        V[1:len, i] .= Vc[i]
    end
    sizein = (Np, Nλ)
    sizeout = (Np, Nl)

    return LinOpIntegration(sizein, sizeout, L, C, V)
end

struct LinOpIntegration{I, O, T} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    rows::Matrix{Int}
    cols::Matrix{Int}
    values::Matrix{T}
end

function LinOpIntegration(sizein::NTuple, sizeout::NTuple, rows::Matrix{Int}, cols::Matrix{Int}, values::Matrix{T}) where {T}
    inputspace = LinOps.CoordinateSpace(sizein)
    outputspace = LinOps.CoordinateSpace(sizeout)
    return LinOpIntegration(inputspace, outputspace, rows, cols, values)
end

function LinOps.apply_!(y, A::LinOpIntegration, x)
    backend = get_backend(x)
    sparse_kernel!(backend)(y, x, A.rows, A.cols, A.values; ndrange = size(A.values, 2))
    return y
end

@kernel function sparse_kernel!(output, input, rows, cols, v)
    i = @index(Global, Linear)
    @inbounds for k in 1:size(output, 2)
        output[i, k] = zero(eltype(output))
    end

    @inbounds for k in 1:size(v, 1)
        output[i, rows[k, i]] += v[k, i] * input[i, cols[k, i]]
    end
end
