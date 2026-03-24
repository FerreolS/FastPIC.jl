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
    valid_lenslets = trues(NLENS)

    # Testing on a small subset for development/runtime.
    test_indices = vcat(collect(1:50), 194, 273, 416, 512, 591, 646, 742, 789, 1083, 1135, 1203)
    valid_lenslets .= false
    valid_lenslets[test_indices] .= true

    calib_params = FastPICParams(; nλ = nλ)
    profiles, lamp_spectra, template, transmission, lλ = calibrate(
        lamp,
        lasers,
        calib_params = calib_params,
        valid_lenslets = valid_lenslets,
    )

    @test length(profiles) == NLENS
    @test length(lamp_spectra) == NLENS
    @test length(template) == length(lλ)

    valid = findall(!isnothing, profiles)
    @test !isempty(valid)
    @test all(isfinite, template)
    @test all(i -> all(isfinite, get_value(transmission[i])), valid)
end
