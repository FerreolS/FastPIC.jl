function export_calib(fitspath, profiles, lamp_spectra, template, transmission, lλ)
    FitsFile(fitspath, "w!") do fits

        # === PRIMARY HDU "TEMPLATE" (Image) === #

        templatehdu = FitsImageHDU(fits, size(template); bitpix=-64)

        templatehdu["EXTNAME"] = ("TEMPLATE", "Common lamp template spectrum")
        templatehdu["HDUNAME"] = ("TEMPLATE", "Common lamp template spectrum")

        templatehdu["CRVAL1"] = (first(lλ), "Coordinate value at reference point")
        templatehdu["CRPIX1"] = (1.0, "Pixel coordinate of reference point")
        templatehdu["CUNIT1"] = ("meter", "Units of coordinate increment and value")
        templatehdu["CTYPE1"] = ("LINEAR", "Coordinate type code")
        templatehdu["CDELT1"] = (step(lλ), "Coordinate increment at reference point")

        write(templatehdu, template)

        # === HDU 2 "PROFILES" (Table) === #

        fi = findfirst(!isnothing, profiles)
        isnothing(fi) && error("every profile is set to `nothing`")

        size_cfwhm = size(profiles[fi].cfwhm)
        size_cx = size(profiles[fi].cx)
        size_spectral_coefs = isnothing(profiles[fi].spectral_coefs) ? 0 :
                              length(profiles[fi].spectral_coefs)

        fi = findfirst(!isnothing, transmission)
        isnothing(fi) && error("every transmission is set to `nothing`")
        length_transmission_factors = length(transmission[fi].value)

        profilehdu = FitsTableHDU(fits,
            "BBOX" => (Int, 4),
            "YCENTER" => Float64,
            "CFWHM" => (Float64, size_cfwhm),
            "CX" => (Float64, size_cx),
            "SPECTRAL_COEFS" => (Float64, size_spectral_coefs),
            "TRANSMISSION" => (Float64, (length_transmission_factors, 2))
        )
        
        profilehdu["EXTNAME"] = ("PROFILES", "parametric model of each spectrum")
        profilehdu["HDUNAME"] = ("PROFILES", "parametric model of each spectrum")
        
        profilehdu["T"] = (string(profiles[fi].type), "numeric type for computations")
        profilehdu["N"] = (ndims(profiles[fi].cfwhm), "ndims of coefs for FWHM variation")
        profilehdu["SIZE_C"] = (size_spectral_coefs, "nb of coefs for wavelength solution")

        write(profilehdu, "BBOX" => stack([
            isnothing(p) ? [0,0,0,0] : [p.bbox[1][1], p.bbox[1][2], p.bbox[2][1], p.bbox[2][2]]
            for p in profiles ]))

        write(profilehdu, "YCENTER" => [ isnothing(p) ? NaN : p.ycenter for p in profiles ])

        write(profilehdu, "CFWHM" => stack([
            isnothing(p) ? fill(NaN, size_cfwhm) : p.cfwhm
            for p in profiles ]))

        write(profilehdu, "CX" => stack([
            isnothing(p) ? fill(NaN, size_cx) : p.cx
            for p in profiles ]))

        write(profilehdu, "SPECTRAL_COEFS" => stack([
            isnothing(p) ? fill(NaN, size_spectral_coefs) : p.spectral_coefs
            for p in profiles ]))

        write(profilehdu, "TRANSMISSION" => stack([
            isnothing(t) ? fill(NaN, length_transmission_factors, 2) : [ t.value ;; t.precision ]
            for t in transmission ]))

        nothing
    end
end

function import_calib(fitspath)
    FitsFile(fitspath) do fits
        
        # === TEMPLATE === #
        template = read(fits["TEMPLATE"])
        
        # === lλ === #
        start_λ = fits["TEMPLATE"]["CRVAL1"].float
        step_λ = fits["TEMPLATE"]["CDELT1"].float
        size_λ = length(template)
        lλ = range(start=start_λ, step=step_λ, length=size_λ)
        
        T = fits["PROFILES"]["T"].string == "Float32" ? Float32 :
            fits["PROFILES"]["T"].string == "Float64" ? Float64 :
            error("cannot parse type")
        N = fits["PROFILES"]["N"].integer
        size_C = fits["PROFILES"]["SIZE_C"].integer
        C = iszero(size_C) ? Nothing : Vector{Float64}
        
        nblenslets = fits["PROFILES"].data_size[1]
        profiles = Vector{Union{Nothing,Profile{T,N,C}}}(undef, nblenslets)
        transmission = Vector{Union{Nothing,WeightedArray{Float64,1}}}(undef, nblenslets)
        P = read(fits["PROFILES"])
        for i in 1:nblenslets
            # === PROFILES === #
            if all(iszero, P["BBOX"][:,i])
                profiles[i] = nothing
            else
                bbox = BoundingBox{Int}(Point(P["BBOX"][1,i], P["BBOX"][2,i]),
                                          Point(P["BBOX"][3,i], P["BBOX"][4,i]))
                ycenter = P["YCENTER"][i]
                cfwhm = P["CFWHM"][ fill(Colon(), N)..., i]
                cx = P["CX"][:,i]
                spectral_coefs = iszero(size_C) ? Nothing : P["SPECTRAL_COEFS"][:,i]
                profiles[i] = Profile(T, bbox, ycenter, cfwhm, cx, spectral_coefs)
            end
            # === TRANSMISSION === #
            if all(isnan, P["TRANSMISSION"][:,:,i])
                transmission[i] = nothing
            else
                transmission[i] = WeightedArray(P["TRANSMISSION"][:,1,i],
                                                P["TRANSMISSION"][:,2,i])
            end
        end

        (profiles, template, transmission, lλ)
    end
end