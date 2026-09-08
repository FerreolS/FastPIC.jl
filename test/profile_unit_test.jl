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
        @test FastPIC.get_bbox(1024, 1024) isa FastPIC.BoundingBox
        @test FastPIC.get_bbox(0, 0) === FastPIC.lenslet_out_of_bounds
    end

    @testset "profile calibration state is explicit" begin
        bbox = FastPIC.BoundingBox(xmin = 1, xmax = 3, ymin = 1, ymax = 4)
        profile = FastPIC.Profile(Float64, bbox, [2.5], [2.0])
        @test !FastPIC.is_calibrated(profile)
        @test_throws ArgumentError FastPIC.get_wavelength(profile)
    end

    @testset "profile constructors share validation path" begin
        bbox = FastPIC.BoundingBox(xmin = 1, xmax = 3, ymin = 1, ymax = 4)
        shorthand = FastPIC.Profile(Float64, bbox, [2.5], [2.0], (3.0, 4.0))
        explicit = FastPIC.Profile(Float64, bbox, mean(axes(bbox, 2)), [2.5], [2.0], nothing, (3.0, 4.0))
        @test shorthand.bbox == explicit.bbox
        @test shorthand.ycenter == explicit.ycenter
        @test shorthand.cfwhm == explicit.cfwhm
        @test shorthand.cx == explicit.cx
        @test shorthand.position == explicit.position
    end
end
