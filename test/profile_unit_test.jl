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
end
