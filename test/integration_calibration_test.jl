@testset "Integration: calibration pipeline (artifact data)" begin

    using AstroFITS
    using WeightedData
    import WeightedData: WeightedArray, get_value
    using LazyArtifacts


    test_data_path = artifact"calibrationtestdata"
    files = joinpath.(test_data_path, ("wave.fits", "specpos.fits"))


    nλ = 3

    wave = openfits(files[1])
    lasers = WeightedArray(read(wave[1])[:, :, 1], read(wave[2])[:, :, 1])
    close(wave)

    specpos = openfits(files[2])
    lamp = WeightedArray(read(specpos[1])[:, :, 1], read(specpos[2])[:, :, 1])
    close(specpos)

    NLENS::Int = 18908

    # Testing on a small subset for development
    valid_lenslets = vcat(194, 273, 416, 512, 591, 646, 742, 789, 1083, 1135, 1203, 1500, 1600, 1700, 1800, 1900, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)


    calib_params = FastPICParams(; nλ = nλ)
    profiles, template, transmission, lλ, lenslet_width = calibrate(
        lamp,
        lasers,
        calib_params = calib_params,
        valid_lenslets = valid_lenslets,
    )

    lamp_spectra = extract_spectra(lamp, profiles; transmission = transmission, restrict = 0, nonnegative = true, refinement_loop = 0)

    λ = lλ[10:2:end]
    PIC = build_PIC_operators(profiles, 300, λ, lenslet_width; pad = 5)


    data = flatten_spectra(lamp_spectra)
    goodpix = data.precision .> 0
    out = PIC' * (data.value .* goodpix)

    @test length(profiles) == length(valid_lenslets)
    @test length(lamp_spectra) == length(profiles)
    @test length(template) == length(lλ)

    valid = findall(!isnothing, profiles)
    @test !isempty(valid)
    @test all(isfinite, template)
    @test all(i -> all(isfinite, get_value(transmission[i])), valid)
end
