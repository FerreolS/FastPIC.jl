using AstroFITS, WeightedData, FastPIC, FITSexplore, FITSHeaders, LinOps, OptimPackNextGen, DifferentiationInterface, Zygote, RobustModels
folder = "/Users/ferreol/Data/RawData/SPHERE/130Elektra/reduced/roots-18"

filedict = fitsexplore(folder)

specposfiles = deepcopy(filedict)
filter_keyword!(specposfiles, Dict("ESO DPR TYPE" => ["SPECPOS,LAMP"]))

wavelampfiles = deepcopy(filedict)
filter_keyword!(wavelampfiles, Dict("ESO DPR TYPE" => ["WAVE,LAMP"]))

(length(specposfiles) != 1  || length(specposfiles) != 1)  && error("single calib")

specposfiles = first(specposfiles)
specposfiles_hdr = specposfiles.second


wavelampfiles = first(wavelampfiles)
wavelampfiles_hdr = wavelampfiles.second

wavelampfiles_hdr["ESO DPR TECH"].value == "IFU" || error("not IFU data")
specposfiles_hdr["ESO DPR TECH"].value == "IFU" || error("not IFU data")

wavelampfiles_hdr["ESO INS2 OPTI2 NAME"].value == specposfiles_hdr["ESO INS2 OPTI2 NAME"].value || error("wavelamp and specpos must have the same mode")


wave = openfits(wavelampfiles.first)
lasers = WeightedArray(read(wave[1]), read(wave[2]))
close(wave)
lasers = mean(lasers, dims = 3)[:, :, 1]

specpos = openfits(specposfiles.first)
lamp = WeightedArray(read(specpos[1]), read(specpos[2]))
close(specpos)
lamp = mean(lamp, dims = 3)[:, :, 1]

nλ = wavelampfiles_hdr["ESO INS2 OPTI2 NAME"].value == "PRI_YJ" ? 3 : 4

calib_params = FastPICParams(; nλ = nλ)
profiles, lamp_spectra, template, transmission, lλ, lenslet_index, lenslet_width = calibrate(
    lamp,
    lasers,
    calib_params = calib_params
)


λ = lλ[3:2:(end - 8)]
Npix = 300
PIC = build_PIC_operators(profiles, Npix, λ, lenslet_width; pad = 5)


objectfiles = deepcopy(filedict)
filter_keyword!(objectfiles, Dict("ESO DPR TYPE" => ["OBJECT"]))

mu = 1.0e-5
G = LinOpGrad(LinOps.inputsize(PIC))
loss = CauchyLoss()
l = Base.Fix1(RobustModels.rho, loss)
r(d, x) = sqrt.(WeightedData.get_precision(d)) .* (x .- WeightedData.get_value(d))

for (filename, header) in objectfiles
    if header["ESO INS2 OPTI2 NAME"].value == wavelampfiles_hdr["ESO INS2 OPTI2 NAME"].value
        ffile = openfits(filename)
        data = WeightedArray(read(ffile[1]), read(ffile[2]))
        close(ffile)
        nframes = header["NAXIS3"].value(Int)
        XX = zeros(inputsize(PIC)..., nframes)
        data_spectra = extract_spectra(data, profiles; transmission = transmission, restrict = 0, nonnegative = true, refinement_loop = 1)
        d = flatten_spectra(data_spectra)
        hdr = readfits(FitsHeader, filename)
        hdr_filtered = filter(!is_structural, hdr)
        for i in 1:nframes
            ff(x) = loglikelihood(d[:, :, i], PIC * x) + mu * sum(abs2, G * x)
            #rr = Base.Fix1(r, d[:, :, i])
            #ff(x) = sum(l.(rr(PIC * x))) + mu * sum(abs2, G * x)
            fg!(x, grad) = DifferentiationInterface.value_and_gradient!(ff, grad, AutoZygote(), x)[1]
            XX[:, :, :, i] = vmlmb(fg!, PIC' * d[:, :, i].value; maxeval = 250, verb = 50, lower = 0.0, xtol = (0.0, 1.0e-9))
        end
        dirname, filename = splitdir(filename)
        newfilename = joinpath(dirname, "reduced", replace(filename, ".fits" => "_reconstructed_$mu.fits"))
        hdr_filtered["HPARAM"] = mu
        hdr_filtered["PIXSCAL"] = 12.25 / (lenslet_width / 2048 * 300)
        writefits!(newfilename, hdr_filtered, XX)
    end
end
lmap = Int32.(FastPIC.build_lenslet_map(profiles; lenslet_width = 1.5, pad = 5))
file = FitsFile(joinpath(folder, "reduced", "lensletmap.fits"), "w!")
hdu = FitsImageHDU(file, lmap)
write(hdu, lmap)
close(file)


#= 
ESO DET ID = 'IFS
ESO DPR TECH = 'IFU     ' / 	
ESO INS2 COMB IFS = 'OBS_H   ' / 
ESO INS2 MODE = 'IFS-H   ' / 
ESO INS2 OPTI2 NAME = 'PRI_YJH ' / 
ESO INS2 OPTI2 NAME = 'PRI_YJ  ' / IFS disperser selector 
=#
