using Test, LazyArtifacts, FastPIC, AstroFITS, WeightedData
using WeightedData: get_value, get_precision

@testset "input output with FITS files" begin
    test_data_path = artifact"calibrationtestdata"
    # to make it work from the REPL by `include`, cd into the "test/" folder first
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

    lmap = FastPIC.build_lenslet_map(profiles; lenslet_width = 1.5)

    mktempdir() do tmpdir
        fitspath = normpath(tmpdir, "fit.fits")
        @test_nowarn FastPIC.export_calib(fitspath, lmap, profiles, template, transmission, lλ)

        local (lmap2, profiles2, template2, transmission2, lλ2)
        @test_nowarn (lmap2, profiles2, template2, transmission2, lλ2) = FastPIC.import_calib(fitspath)

        @test lmap == lmap2
        @test size(profiles) == size(profiles2)
        for i in eachindex(profiles, profiles2)
            @test profiles[i].type == profiles2[i].type
            @test profiles[i].bbox == profiles2[i].bbox
            @test profiles[i].ycenter == profiles2[i].ycenter
            @test profiles[i].cfwhm == profiles2[i].cfwhm
            @test profiles[i].cx == profiles2[i].cx
            @test profiles[i].spectral_coefs == profiles2[i].spectral_coefs
            @test profiles[i].position == profiles2[i].position
        end
        @test template ≈ template2
        @test size(transmission) == size(transmission2)
        for i in eachindex(transmission, transmission2)
            @test get_value(transmission[i]) ≈ get_value(transmission2[i])
            @test get_precision(transmission[i]) ≈ get_precision(transmission2[i])
        end
        @test lλ ≈ lλ2
    end
end
