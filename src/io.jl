function export_calib(fitspath, profiles, template, transmission, lλ, lenslet_width, lenslet_θ)
    return FitsFile(fitspath, "w!") do fits

        # === PRIMARY HDU "LMAP" (Image) === #
        lmap = get_lensletmap(profiles)
        lmaphdu = FitsImageHDU(fits, size(lmap); bitpix = 64) # eltype=Int64
        lmaphdu["EXTNAME"] = ("LMAP", "lenslets map")
        lmaphdu["HDUNAME"] = ("LMAP", "lenslets map")
        lmaphdu["LENSDIST"] = (lenslet_width, "distance between lenslet centers in pixels")
        lmaphdu["LENSROT"] = (rem(lenslet_θ / π * 180, 60.0, RoundNearest), "rotation angle of the lenslet grid in degrees")

        write(lmaphdu, lmap)

        # === HDU 2 "TEMPLATE" (Image) === #

        templatehdu = FitsImageHDU(fits, size(template); bitpix = -64) # eltype=Float64

        templatehdu["EXTNAME"] = ("TEMPLATE", "Common lamp template spectrum")
        templatehdu["HDUNAME"] = ("TEMPLATE", "Common lamp template spectrum")

        templatehdu["CRVAL1"] = (first(lλ), "Coordinate value at reference point")
        templatehdu["CRPIX1"] = (1.0, "Pixel coordinate of reference point")
        templatehdu["CUNIT1"] = ("meter", "Units of coordinate increment and value")
        templatehdu["CTYPE1"] = ("LINEAR", "Coordinate type code")
        templatehdu["CDELT1"] = (step(lλ), "Coordinate increment at reference point")

        write(templatehdu, template)

        # === HDU 3 "PROFILES" (Table) === #

        fipr = findfirst(!isnothing, profiles)
        isnothing(fipr) && error("every profile is set to `nothing`")

        size_cfwhm = size(profiles[fipr].cfwhm)
        size_cx = size(profiles[fipr].cx)
        size_spectral_coefs = isnothing(profiles[fipr].spectral_coefs) ? 0 :
            length(profiles[fipr].spectral_coefs)

        fitr = findfirst(!isnothing, transmission)
        isnothing(fitr) && error("every transmission is set to `nothing`")
        length_transmission_factors = length(transmission[fitr].value)

        profilehdu = FitsTableHDU(
            fits,
            "BBOX" => (Int, 4),
            "YCENTER" => Float64,
            "CFWHM" => (Float64, size_cfwhm),
            "CX" => (Float64, size_cx),
            "SPECTRAL_COEFS" => (Float64, size_spectral_coefs),
            "POSITION" => (Float64, 2),
            "TRANSMISSION" => (Float64, (length_transmission_factors, 2))
        )

        profilehdu["EXTNAME"] = ("PROFILES", "parametric model of each spectrum")
        profilehdu["HDUNAME"] = ("PROFILES", "parametric model of each spectrum")

        profilehdu["T"] = (string(profiles[fipr].type), "numeric type for computations")
        profilehdu["N"] = (ndims(profiles[fipr].cfwhm), "ndims of coefs for FWHM variation")
        profilehdu["SIZE_C"] = (size_spectral_coefs, "nb of coefs for wavelength solution")

        write(
            profilehdu, "BBOX" => stack(
                [
                    isnothing(p) ? [0, 0, 0, 0] : [p.bbox[1][1], p.bbox[1][2], p.bbox[2][1], p.bbox[2][2]]
                        for p in profiles
                ]
            )
        )

        write(profilehdu, "YCENTER" => [ isnothing(p) ? NaN : p.ycenter for p in profiles ])

        write(
            profilehdu, "CFWHM" => stack(
                [
                    isnothing(p) ? fill(NaN, size_cfwhm) : p.cfwhm
                        for p in profiles
                ]
            )
        )

        write(
            profilehdu, "CX" => stack(
                [
                    isnothing(p) ? fill(NaN, size_cx) : p.cx
                        for p in profiles
                ]
            )
        )

        write(
            profilehdu, "SPECTRAL_COEFS" => stack(
                [
                    isnothing(p) ? fill(NaN, size_spectral_coefs) : p.spectral_coefs
                        for p in profiles
                ]
            )
        )

        write(
            profilehdu, "POSITION" => stack(
                [
                    isnothing(p) ? [NaN64, NaN64] : [p.position[1], p.position[2]]
                        for p in profiles
                ]
            )
        )

        write(
            profilehdu, "TRANSMISSION" => stack(
                [
                    isnothing(t) ? fill(NaN, length_transmission_factors, 2) : [ t.value ;; t.precision ]
                        for t in transmission
                ]
            )
        )

        nothing
    end
end

function import_calib(fitspath)
    return FitsFile(fitspath) do fits

        # === LMAP === #
        lmap = read(fits["LMAP"])

        # === TEMPLATE === #
        template = read(fits["TEMPLATE"])

        # === lλ === #
        start_λ = fits["TEMPLATE"]["CRVAL1"].float
        step_λ = fits["TEMPLATE"]["CDELT1"].float
        size_λ = length(template)
        lλ = range(start = start_λ, step = step_λ, length = size_λ)

        T = fits["PROFILES"]["T"].string == "Float32" ? Float32 :
            fits["PROFILES"]["T"].string == "Float64" ? Float64 :
            error("cannot parse type")
        N = fits["PROFILES"]["N"].integer
        size_C = fits["PROFILES"]["SIZE_C"].integer
        C = iszero(size_C) ? Nothing : Vector{Float64}

        nblenslets = fits["PROFILES"].data_size[1]
        profiles = Vector{Union{Nothing, Profile{T, N, C}}}(undef, nblenslets)
        transmission = Vector{Union{Nothing, WeightedArray{Float64, 1}}}(undef, nblenslets)
        P = read(fits["PROFILES"])
        for i in 1:nblenslets
            # === PROFILES === #
            if all(iszero, P["BBOX"][:, i])
                profiles[i] = nothing
            else
                bbox = BoundingBox{Int}(
                    Point(P["BBOX"][1, i], P["BBOX"][2, i]),
                    Point(P["BBOX"][3, i], P["BBOX"][4, i])
                )
                ycenter = P["YCENTER"][i]
                cfwhm = P["CFWHM"][fill(Colon(), N)..., i]
                cx = P["CX"][:, i]
                spectral_coefs = iszero(size_C) ? Nothing : P["SPECTRAL_COEFS"][:, i]
                position = (P["POSITION"][1, i], P["POSITION"][2, i])
                profiles[i] = Profile(T, bbox, ycenter, cfwhm, cx, spectral_coefs, position)
            end
            # === TRANSMISSION === #
            if all(isnan, P["TRANSMISSION"][:, :, i])
                transmission[i] = nothing
            else
                transmission[i] = WeightedArray(
                    P["TRANSMISSION"][:, 1, i],
                    P["TRANSMISSION"][:, 2, i]
                )
            end
        end

        (lmap, profiles, template, transmission, lλ)
    end
end
