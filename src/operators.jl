using NFFT,
    Unitful,
    UnitfulAngles,
    FFTW,
    Bessels


function make_IC_operator(
        profiles;
        Npixels = 300,
        lenslets_position = :theoretical,
        pixsize = 5.75u"mas",
        diameter = 8u"m",
        onsky_lenslets_sep = 12.25u"mas",
        λmin = 0.96u"μm",
        oncamera_lenslets_sep = 8
    )

    lenslet_conv = lenslet_convolution(Npixels, onsky_lenslets_sep, pixsize)
    return lenslet_conv
end
function get_pupil_mask(Npixels, diameter, pixsize, λmin)
    cutoff = (pixsize * (((diameter / λmin) |> NoUnits) * 1.0u"rad")) |> NoUnits
    return sqrt.(fftfreq(Npixels) .^ 2 .+ (fftfreq(Npixels) .^ 2)') .<= cutoff
end

function lenslet_convolution(Npixels, lenslets_sep, pixsize)
    lenslet_radius = lenslets_sep / pixsize / 2
    uv = sqrt.(fftfreq(Npixels) .^ 2 .+ (fftfreq(Npixels) .^ 2)')
    p = (besselj1.(2π .* uv .* lenslet_radius) ./ (uv .* lenslet_radius))
    p[1, 1] = π
    return p .* ((lenslet_radius)^2)
end

struct LinOpNFFT{
        I, O,
        F <: NFFTPlan,     # type of forward plan
    } <: AbstractLinOp{I, O}
    inputspace::I
    outputspace::O
    plan::F             # plan for forward transform
end
lenslets_sep = 12.25u"mas"
pixsize = 5.75u"mas"
T = Float32
N = 300
Ns = (N, N)
d = [ laser_spectra[i][20].value for i in findall(!isnothing, profiles)]
l = Float32.(lenslet_convolution(N, lenslets_sep, pixsize));

cx = [profiles[i].cx[1] for i in findall(!isnothing, profiles)]
cy = [profiles[i].ycenter for i in findall(!isnothing, profiles)]
Np = length(d)
plan_nufft = PlanNUFFT(T, Ns; m = HalfSupport(4))

points = (T.(cx), T.(cy))
set_points!(plan_nufft, points)
vp = Array{T}(undef, Np)
r2 = T.(gridded_weighted150[:, :, 20].value)
exec_type2!(vp, plan_nufft, l[1:76, :] .* rfft(r2))
