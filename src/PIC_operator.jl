function build_PIC_operators(profiles, Npix, λ, lenslet_width; T=Float64 )

    sz = (Npix, Npix, length(λ))
    cx = [ p.position[1] for p in profiles] ./ 2048 .* 2π
    cy = [ p.position[2] for p in profiles] ./ 2048 .* 2π
    points = (cx, cy)
    lenslet_radius = lenslet_width / 2048 * Npix / 2

    if T <: Complex
        r2c = false
    else
        r2c = true
    end

    mtf = compute_airy_mtf(Npix, lenslet_radius; normalize = true, r2c = r2c)
    D = LinOpDiag(mtf)
    sz2 = (round(Int, sz[1] / 2) + 1, sz[2:end]...)
    C = LinOpMapslice(sz2, D, [1, 2]) * LinOpDFT(T, sz, dims = [1, 2])
    NF = LinOpNFFT(T, sz[1:2], points; sort_points = True())
    II = LinOpMapslice(sz2, NF', [1, 2])

    MI = [ FastPIC.build_sparse_interpolation_integration_matrix( get_precision( T) ,λ, profile) for profile in profiles]

    P = LinOpMapslice(outputsize(II), MI, 2)
    PIC = P * II * C * UniformScaling(T(2 / (Npix^2)))

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
