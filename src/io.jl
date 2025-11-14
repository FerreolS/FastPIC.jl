function export_calib(fitspath, profiles, lamp_spectra, template, transmission, lλ)
    FitsFile(fitspath, "w!") do fits

        # === PRIMARY HDU "TEMPLATE" (Image) === #

        templatehdu = FitsImageHDU(fits, (length(template), 1); bitpix=-64)

        templatehdu["EXTNAME"] = ("TEMPLATE", "Common lamp template spectrum")
        templatehdu["HDUNAME"] = ("TEMPLATE", "Common lamp template spectrum")

        write(templatehdu, reshape(template, Val(2))) # 2 dims so DS9 can open it

        # === HDU 2 "PROFILES" (Table) === #

        fi = findfirst(!isnothing, profiles)
        isnothing(fi) && error("every profile is set to `nothing`")

        size_cfwhm = size(profiles[fi].cfwhm)
        size_cx = size(profiles[fi].cx)
        size_spectral_coefs = size(profiles[fi].spectral_coefs)

        profilehdu = FitsTableHDU(fits,
            "BBOX" => (Int, 4),
            "YCENTER" => Float64,
            "CFWHM" => (Float64, size_cfwhm...),
            "CX" => (Float64, size_cx...),
            "SPECTRAL_COEFS" => (Float64, size_spectral_coefs...)
        )
        
        profilehdu["EXTNAME"] = ("PROFILES", "parametric model of each spectrum")
        profilehdu["HDUNAME"] = ("PROFILES", "parametric model of each spectrum")

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

        # === HDU 3 "TRANSMISSION" (Image) === #

        fi = findfirst(!isnothing, transmission)
        length_factors = length(transmission[fi])

        transmissionarray = Array{Float64,3}(undef, length(transmission), length_factors, 2)

        for i in 1:length(transmission)
            if isnothing(transmission[i])
                transmissionarray[i,:,:] .= NaN
            else
                transmissionarray[i,:,1] .= transmission[i].args[1]
                transmissionarray[i,:,2] .= transmission[i].args[2]
            end
        end

        transmissionhdu = FitsImageHDU(fits, size(transmissionarray); bitpix=-64)
        
        transmissionhdu["EXTNAME"] = ("TRANSMISSION", "Transmission factors for each lenslet")
        transmissionhdu["HDUNAME"] = ("TRANSMISSION", "Transmission factors for each lenslet")
        
        write(transmissionhdu, transmissionarray)
        
        nothing
    end
end