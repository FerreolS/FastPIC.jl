@testset "Unit: profile helpers" begin
    using WeightedData
    import WeightedData: WeightedArray

    @testset "filter_spectra_outliers! flags strong outlier" begin
        spectra = [
            WeightedArray([1.0, 1.0, 1.0], ones(3)),
            WeightedArray([1.0, 100.0, 1.0], ones(3)),
            WeightedArray([1.0, 1.0, 1.0], ones(3)),
        ]

        FastPIC.filter_spectra_outliers!(spectra; threshold = 1)

        @test spectra[2].value[2] == 0
        @test spectra[2].precision[2] == 0
        @test spectra[1].value[2] == 1.0
        @test spectra[3].value[2] == 1.0
    end

    @testset "filter_spectra_outliers returns copy" begin
        source = [
            WeightedArray([1.0, 100.0, 1.0], ones(3)),
            WeightedArray([1.0, 1.0, 1.0], ones(3)),
            WeightedArray([1.0, 1.0, 1.0], ones(3)),
        ]

        result = FastPIC.filter_spectra_outliers(source; threshold = 1)

        @test result !== source
        @test source[1].value[2] == 100.0
        @test result[1].value[2] == 0
        @test result[1].precision[2] == 0
    end

    @testset "get_bbox reports geometry status" begin
        bbox = FastPIC.get_bbox(1024, 1024)
        @test bbox isa FastPIC.ProfileBBox
        @test size(bbox) == (FastPIC.BboxParams().BBOX_WIDTH, FastPIC.BboxParams().BBOX_HEIGHT)
        @test FastPIC.get_bbox(0, 0) === FastPIC.lenslet_out_of_bounds

        limits = CartesianIndices((1:10, 1:10))
        @test FastPIC.grow_bbox(CartesianIndices((5:7, 4:6)), 2, 1, limits) ==
            CartesianIndices((3:9, 3:7))
        off_origin = CartesianIndices((10:20, 100:110))
        off_origin_limits = CartesianIndices((1:50, 1:150))
        @test FastPIC. grow_bbox(off_origin, 2, 3, off_origin_limits) ==
            CartesianIndices((8:22, 97:113))
    end

    @testset "profile calibration state is explicit" begin
        bbox = CartesianIndices((1:3, 1:4))
        profile = FastPIC.Profile(Float64, bbox, [2.5], [2.0])
        @test profile.bbox isa CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}
        @test profile.bbox == CartesianIndices((1:3, 1:4))
        @test profile() == profile(bbox)
        @test !FastPIC.is_calibrated(profile)
        @test_throws ArgumentError FastPIC.get_wavelength(profile)
    end

    @testset "profile constructors share validation path" begin
        bbox = CartesianIndices((1:3, 1:4))
        shorthand = FastPIC.Profile(Float64, bbox, [2.5], [2.0], (3.0, 4.0))
        explicit = FastPIC.Profile(Float64, bbox, mean(FastPIC.bbox_ranges(bbox, 2)), [2.5], [2.0], nothing, (3.0, 4.0))
        @test shorthand.bbox == explicit.bbox
        @test shorthand.ycenter == explicit.ycenter
        @test shorthand.cfwhm == explicit.cfwhm
        @test shorthand.cx == explicit.cx
        @test shorthand.position == explicit.position
    end
end
