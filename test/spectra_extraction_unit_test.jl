@testset "Unit: spectra extraction" begin
    using WeightedData
    import WeightedData: WeightedArray, get_value, get_precision

    bbox1 = FastPIC.BoundingBox(xmin = 1, xmax = 3, ymin = 1, ymax = 4)
    bbox2 = FastPIC.BoundingBox(xmin = 4, xmax = 6, ymin = 1, ymax = 4)

    p1 = FastPIC.Profile(Float64, bbox1, [2.5], [2.0])
    p2 = FastPIC.Profile(Float64, bbox2, [2.5], [5.0])

    m1 = p1()
    m2 = p2()

    α1 = [1.0, 2.0, 3.0, 4.0]
    α2 = [2.0, 4.0, 6.0, 8.0]

    values = zeros(6, 4)
    values[1:3, :] .= m1 .* reshape(α1, 1, :)
    values[4:6, :] .= m2 .* reshape(α2, 1, :)
    precisions = ones(6, 4)
    data = WeightedArray(values, precisions)

    @testset "extract_spectrum recovers known amplitudes" begin
        s1 = FastPIC.extract_spectrum(data, p1)
        @test get_value(s1) ≈ α1 atol = 1.0e-10
        @test get_precision(s1) ≈ vec(sum(m1 .^ 2; dims = 1)) atol = 1.0e-10
    end

    @testset "extract_spectrum nonnegative masks negative bins" begin
        αneg = [-1.0, 2.0, -0.5, 4.0]
        vals = zeros(3, 4)
        vals[:, :] .= m1 .* reshape(αneg, 1, :)
        dneg = WeightedArray(vals, ones(3, 4))

        sneg = FastPIC.extract_spectrum(dneg, p1; inbbox = true, nonnegative = true)
        @test get_value(sneg) == [0.0, 2.0, 0.0, 4.0]
        @test get_precision(sneg)[1] == 0
        @test get_precision(sneg)[3] == 0
    end

    @testset "extract_spectrum restrict truncates model" begin
        srestricted = FastPIC.extract_spectrum(data, p1; restrict = 2.0)
        @test all(iszero, get_value(srestricted))
        @test all(iszero, get_precision(srestricted))
    end

    @testset "extract_spectra handles valid profiles and scalar transmission" begin
        profiles = [p1, p2]
        tr = [2.0, 1.0, 4.0]
        spectra = FastPIC.extract_spectra(data, profiles; transmission = tr[1:2], nonnegative = false, ntasks = 1)

        @test get_value(spectra[1]) ≈ α1 ./ 2 atol = 1.0e-10
        @test get_value(spectra[2]) ≈ α2 atol = 1.0e-10
    end


    @testset "flatten_spectra accepts weighted vectors" begin
        spectra = [WeightedArray([1.0, 2.0], [4.0, 9.0])]
        flattened = FastPIC.flatten_spectra(spectra)
        @test size(flattened) == (1, 2)
        @test get_value(flattened) == [1.0 2.0]
        @test get_precision(flattened) == [4.0 9.0]
    end
end
