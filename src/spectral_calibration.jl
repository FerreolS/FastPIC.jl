"""
    LaserModel

Model for laser emission lines used in spectral calibration.

# Fields
- `position::Vector{Float64}`: Pixel positions of laser lines
- `fwhm::Vector{Float64}`: Full width at half maximum of each laser line in pixels

# Examples
```julia
laser_model = LaserModel([10.5, 25.3, 40.1], [2.0, 2.1, 1.9])
```
"""
struct LaserModel
    #   wavelength::Vector{Float64}
    position::Vector{Float64}
    fwhm::Vector{Float64}
end
#trainable(x::LaserModel) = (; position=x.position, fwhm=x.fwhm)

LaserModel(position::AbstractVector, fwhm::AbstractVector) =
    LaserModel(collect(position), collect(fwhm))

"""
    compute_laser_images(laser::LaserModel, idx::AbstractVector)

Generate 1D Gaussian profiles for laser lines at specified pixel positions.

# Arguments
- `laser::LaserModel`: Laser model containing positions and FWHM values
- `idx::AbstractVector`: Pixel indices where to evaluate the profiles

# Returns
- `Matrix{Float64}`: 2D array where each column is a Gaussian profile for one laser line
"""
function compute_laser_images(
        (; position, fwhm)::LaserModel,
        idx::AbstractVector
    )
    fwhm2sigma = 1 / (2 * sqrt(2 * log(2)))
    fw = 2 .* (fwhm .* fwhm2sigma)
    return exp.(-((idx .- reshape(position, 1, :)) ./ reshape(fw, 1, :)) .^ 2)
end

"""
    compute_lasers_amplitudes(::Val{N}, model::Array{T,2}, data::WeightedArray) where {T<:Real, N}

Compute optimal amplitudes for laser lines using weighted least squares.

Solves the linear system `A * amplitudes = b` where A is the cross-correlation matrix
of the laser profiles weighted by data precision.

# Arguments
- `::Val{N}`: Number of laser lines (compile-time constant)
- `model::Array{T,2}`: Matrix of laser profiles (pixels × lines)
- `data::WeightedArray`: Observed data with uncertainties

# Returns
- `StaticVector{N,T}`: Optimal amplitudes for each laser line
"""
function compute_lasers_amplitudes(
        ::Val{N},
        model::Array{T, 2},
        (; value, precision)::WeightedArray
    ) where {T <: Real, N}

    A = @MMatrix zeros(T, N, N)
    b = @MVector zeros(T, N)

    @inbounds for index in 1:N
        mw = model[:, index] .* precision
        b[index] = mw' * value
        A[index, index] = mw' * model[:, index]
        for i in 1:(index - 1)
            A[i, index] = A[index, i] = mw' * model[:, i]
        end
    end

    return inv(A) * b

end

"""
    laser_cost(data::WeightedArray, lasers::LaserModel)

Compute the negative log-likelihood cost for laser model fitting.
The model is composed of Gaussian profiles at the laser positions plus a constant baseline.

# Arguments
- `data::WeightedArray`: Observed laser spectrum data
- `lasers::LaserModel`: Current laser model parameters

# Returns
- `Float64`: Negative log-likelihood value (lower is better fit)
"""
function laser_cost(
        data::WeightedArray,
        lasers::LaserModel
    )
    images = hcat(compute_laser_images(lasers, axes(data, 1)), ones(length(data)))
    amplitude = compute_lasers_amplitudes(Val(size(images, 2)), images, data)
    model = images * amplitude
    return likelihood(data, model)
end


"""
    fit_laser(data::WeightedArray, laser::LaserModel)

Fit laser line positions and widths to observed data using NEWUOA optimization.

# Arguments
- `data::WeightedArray`: Observed laser spectrum with uncertainties
- `laser::LaserModel`: Initial guess for laser parameters

# Returns
- `LaserModel`: Optimized laser model with fitted positions and FWHM values

# Examples
```julia
initial_model = LaserModel([10.0, 25.0, 40.0], [2.0, 2.0, 2.0])
fitted_model = fit_laser(laser_data, initial_model)
```
"""
function fit_laser(
        data::WeightedArray,
        laser::LaserModel
    )
    vec, re = Optimisers.destructure(laser)
    f(x) = laser_cost(data, re(x))
    Newuoa.optimize!(f, vec, 1.0e-5, 1.0e-15; check = false, maxeval = 10_000, verbose = 0)
    return re(vec)
end

"""
    get_laser_precision(laser::LaserModel, data::WeightedArray)

Compute the Fisher information matrix for laser line parameter uncertainties.

# Arguments
- `laser::LaserModel`: Fitted laser model
- `data::WeightedArray`: Observed data used for fitting

# Returns
- `StaticMatrix`: Precision matrix (inverse covariance) for laser line amplitudes
"""
get_laser_precision(laser::LaserModel, data::WeightedArray) = get_laser_precision(Val(length(laser.position)), laser, data)

function get_laser_precision(
        ::Val{N},
        laser::LaserModel,
        (; value, precision)::WeightedArray
    ) where {N}
    images = compute_laser_images(laser, axes(value, 1))
    W = @MMatrix zeros(Float64, N, N)

    @inbounds for index in 1:N
        mw = images[:, index] .* precision
        W[index, index] = mw' * images[:, index]
        for i in 1:(index - 1)
            W[i, index] = W[index, i] = mw' * images[:, i]
        end
    end
    return W
end


"""
    laser_calibration(order, ref, lasers_λs, laser_positions, Wpos)

Compute polynomial wavelength calibration coefficients from laser line data.

Fits a polynomial of the form: `λ = Σ coefs[i] * ((pixel - ref)/ref)^(i-1)`

# Arguments
- `order::Int`: Polynomial order for wavelength solution
- `ref::Float64`: Reference pixel position
- `lasers_λs::Vector{Float64}`: Known wavelengths of laser lines
- `laser_positions::Vector{Float64}`: Fitted pixel positions of laser lines
- `Wpos::Matrix{Float64}`: Precision matrix for laser positions

# Returns
- `Vector{Float64}`: Polynomial coefficients for wavelength calibration
"""
laser_calibration(order, ref, lasers_λs, laser_positions, Wpos) = laser_calibration(Val(order), Val(length(laser_positions)), ref, lasers_λs, laser_positions, Wpos)

function laser_calibration(::Val{order}, ::Val{lines}, ref, lasers_λs, laser_positions, Wpos) where {order, lines}
    A = MMatrix{lines, order + 1}(((laser_positions .- ref) ./ ref) .^ reshape(0:order, 1, :))
    coefs = inv(A' * Wpos * A) * A' * Wpos * lasers_λs
    return coefs
end

"""
    build_λrange(λs; superres=1)

Build a common wavelength grid covering all input wavelength ranges.

# Arguments
- `λs`: Wavelength arrays (Matrix or Vector of Vectors)
- `superres::Real=1`: Super-resolution factor for sampling density

# Returns
- `StepRangeLen`: Common wavelength grid with uniform sampling

# Examples
```julia
λ_grid = build_λrange(wavelength_arrays; superres=2.0)
```
"""
function build_λrange(λs::AbstractMatrix{<:Real}; superres = 1)
    nb_el = round(Int, size(λs, 1) * superres)
    blue = λs[1, :]
    red = λs[end, :]
    return range(start = minimum(blue), stop = maximum(red), step = median((red .- blue)) ./ nb_el)
end

function build_λrange(λs::Vector{<:Union{Nothing, Vector{Float64}}}; superres = 1)
    λ = filter(!isnothing, λs)
    nb_el = round(Int, maximum(length.(λ)) * superres)
    blue = minimum.(λ)
    red = maximum.(λ)
    return range(start = minimum(blue), stop = maximum(red), step = median((red .- blue)) ./ nb_el)
end


"""
    estimate_template(λ, coefs, spectra; regul=1)

Estimate a common spectral template from multiple observed spectra.

Uses Tikhonov regularization to solve for a smooth template spectrum that best
explains all input spectra when convolved with their respective wavelength solutions.

# Arguments
- `λ::AbstractVector`: Common wavelength grid
- `coefs::Vector`: Wavelength calibration coefficients for each spectrum
- `spectra::Vector`: Observed spectra with uncertainties
- `regul::Real=1`: Tikhonov regularization parameter

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: Template spectrum and transmission factors
"""
function estimate_template(
        profiles::AbstractVector{<:Union{Nothing, Profile}},
        λ,
        coefs,
        spectra;
        regul = 1
    )
    nλ = length(λ)
    valid_lenslets = map(!isnothing, profiles)
    transmission = zeros(Float64, length(valid_lenslets))
    MI = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(valid_lenslets))
    if regul == 0
        A = zeros(Float64, nλ, nλ)
    else
        diagA = 2 * regul * ones(Float64, nλ)
        diagA[1] = regul
        diagA[end] = regul
        A = Array(BandedMatrix((0 => diagA, 1 => -regul * ones(nλ - 1), -1 => -regul * ones(nλ - 1)), (nλ, nλ)))
    end
    b = zeros(Float64, nλ)
    foreach(findall(valid_lenslets)) do idx
        (; value, precision) = spectra[idx]
        profile_wavelength = get_wavelength(coefs[idx], profiles[idx].ycenter - profiles[idx].bbox.ymin, axes(value, 1))
        MI[idx] = build_sparse_interpolation_integration_matrix(λ, get_lower_uppersamples(profile_wavelength)...)
        b .+= Array(MI[idx]' * (precision .* value))
        A .+= Array(MI[idx]' * (precision .* MI[idx]))
    end


    template = A \ b

    OhMyThreads.tforeach(findall(valid_lenslets)) do idx
        (; value, precision) = spectra[idx]
        m = (MI[idx] * template)
        transmission[idx] = sum((mp = m .* precision) .* value) / sum(m .* mp)
    end

    transmission .*= 1 ./ median(transmission[findall(valid_lenslets)])

    return template, transmission
end


"""
    lamp_model(λ, spectrum_template, template_wavelength)

Project a high-resolution lamp template onto a given lenslet wavelength grid.
It builds a sparse matrix that represents the interpolation and integration process, then applies it to the template spectrum.

# Arguments
- `λ::AbstractVector{<:Real}`: The target wavelength grid (e.g., the wavelengths corresponding to lenslet pixels).
- `spectrum_template::AbstractVector{<:Real}`: The intensity values of the high-resolution template lamp spectrum.
- `template_wavelength::AbstractVector{<:Real}`: The wavelength grid corresponding to `spectrum_template`.

# Returns
- `AbstractVector{<:Real}`: The modeled lamp spectrum on the wavelength grid `λ`.

# See Also
- [`get_lower_uppersamples`](@ref)
- [`build_sparse_interpolation_integration_matrix`](@ref)
"""
function lamp_model(λ, spectrum_template, template_wavelength)
    lo, up = get_lower_uppersamples(λ)
    model = build_sparse_interpolation_integration_matrix(template_wavelength, lo, up) * spectrum_template
    return model
end


"""
    laser_model(λ, fwhm_pixels, lasers_λs, data)

Build a model of the lasers that fits the  measured data.

The function models each laser peak (e.g., as a Gaussian) on the lenslet's wavelength grid `λ`. 
It also includes a constant background term. The amplitudes of the laser peaks and the background are 
determined by a linear least-squares fit to the provided `data`.

# Arguments
- `λ::AbstractVector{<:Real}`: The wavelength grid of the spectrometer.
- `fwhm_pixels::Union{Real, AbstractVector{<:Real}}`: The Full Width at Half Maximum (FWHM) of the laser peaks, expressed in units of pixels. Can be a single value for all lasers or a vector with a value for each laser.
- `lasers_λs::AbstractVector{<:Real}`: A vector containing the central wavelengths of the lasers to be modeled.
- `data::AbstractVector{<:Real}`: The measured spectral data to which the model will be fitted.

# Returns
- `AbstractVector{<:Real}`: The best-fit model of the laser spectra, including the background, on the wavelength grid `λ`.

# See Also
- [`LaserModel`](@ref)
- [`compute_laser_images`](@ref)
- [`compute_lasers_amplitudes`](@ref)
"""
function laser_model(λ, fwhm_pixels, lasers_λs, data)
    idx = max.(2, [searchsortedlast(λ, l) for l in lasers_λs])
    las = LaserModel(lasers_λs, fwhm_pixels .* (λ[idx] .- λ[idx .- 1]))
    images = hcat(compute_laser_images(las, λ), ones(length(λ)))
    amplitude = compute_lasers_amplitudes(Val(length(lasers_λs) + 1), images, data)
    return images * amplitude
end

"""
    spectral_refinement(coefs, lamp, lamp_template, wavelength, reference_pixel, lasers_λs, fwhm_pixels, laser)

Refine wavelength calibration coefficients using both lamp and laser data.

Optimizes wavelength solution to minimize combined residuals from lamp template
and laser line fits.

# Arguments
- `coefs::Vector{Float64}`: Initial wavelength calibration coefficients
- `lamp::WeightedArray`: Observed lamp spectrum
- `lamp_template::Vector{Float64}`: Reference lamp template
- `wavelength::AbstractVector`: Template wavelength grid
- `reference_pixel::Float64`: Reference pixel position
- `lasers_λs::Vector{Float64}`: Known laser wavelengths
- `fwhm_pixels::Vector{Float64}`: Laser line widths in pixels
- `laser::WeightedArray`: Observed laser spectrum

# Returns
- `Vector{Float64}`: Refined wavelength calibration coefficients
"""
function spectral_refinement(coefs, lamp, lamp_template, wavelength, reference_pixel, lasers_λs, fwhm_pixels, laser)
    function loss(x)
        wvlngth = get_wavelength(x, reference_pixel, axes(lamp, 1))
        lamp_spectrum = lamp_model(wvlngth, lamp_template, wavelength)
        laser_spectrum = laser_model(wvlngth, fwhm_pixels, lasers_λs, laser)
        return likelihood(ScaledL2Loss(), lamp, lamp_spectrum) + likelihood(laser, laser_spectrum)
    end
    #scale = 1e-7 .* vcat(10. .^ (-(1:length(coefs))))
    scale = 1.0e-8 .* ones(length(coefs))
    Newuoa.optimize!(loss, coefs, 1, 1.0e-9; scale = scale, check = false, maxeval = 10_000, verbose = 0)
    return coefs
end

"""
    recalibrate_wavelengths(λ, coefs, order, lamp_spectra, laser_spectra, lasers_λs, lasers_model; kwargs...)

Iteratively refine wavelength calibrations for all valid lenslets.

Performs iterative refinement where each iteration:
1. Estimates a common lamp template from all spectra
2. Refines individual wavelength solutions using template + laser constraints
3. Updates the template with improved calibrations

# Arguments
- `λ::AbstractVector`: Common wavelength grid
- `coefs::Vector`: Initial wavelength calibration coefficients
- `order::Int`: Final polynomial order for wavelength solutions
- `lamp_spectra::Vector`: Observed lamp spectra
- `laser_spectra::Vector`: Observed laser spectra  
- `lasers_λs::Vector{Float64}`: Known laser wavelengths
- `lasers_model::Vector`: Fitted laser models

# Keyword Arguments
- `verbose::Bool=false`: Enable progress reporting
- `ntasks::Int=4*Threads.nthreads()`: Number of parallel tasks
- `regul::Real=1`: Tikhonov regularization parameter
- `loop::Int=2`: Number of refinement iterations

# Returns
- `Tuple{Vector, Vector{Float64}, Vector{Float64}}`: Refined coefficients, template, transmission
"""
function recalibrate_wavelengths(
        profiles::AbstractVector{<:Union{Nothing, Profile}},
        λ,
        coefs,
        order,
        lamp_spectra,
        laser_spectra,
        lasers_λs,
        lasers_model;
        verbose = false,
        ntasks = Threads.nthreads() * 4,
        regul = 1,
        loop = 2 # TODO put in calib_params
    )
    valid_lenslets = map(!isnothing, profiles)

    template, transmission = estimate_template(profiles, λ, coefs, lamp_spectra; regul = regul)

    new_coefs = Vector{Union{Nothing, Vector{Float64}}}(undef, length(coefs))
    fill!(new_coefs, nothing)

    progressbar = verbose ? Progress(sum(valid_lenslets) * loop; showspeed = true, desc = "Spectral recalibration $loop loops") : nothing

    for _ in 1:loop
        @localize coefs @localize template  OhMyThreads.tforeach(findall(valid_lenslets); ntasks = ntasks) do i
            #foreach(findall(valid_lenslets)) do i
            if (order + 1) > length(coefs[i])
                coef = vcat(coefs[i], zeros(order - length(coefs[i]) + 1))
            else
                coef = copy(coefs[i])
            end
            try
                new_coefs[i] = spectral_refinement(coef, lamp_spectra[i], template, λ, profiles[i].ycenter - profiles[i].bbox.ymin, lasers_λs, lasers_model[i].fwhm, laser_spectra[i])
            catch e
                @debug "Spectral refinement failed for lenslet $i: $e"
                valid_lenslets[i] = false
                profiles[i] = nothing
            end
            verbose && next!(progressbar)
        end
        coefs = copy(new_coefs)

        template, transmission = estimate_template(profiles, λ, coefs, lamp_spectra)

    end
    verbose && (finish!(progressbar))
    return coefs, template, transmission
end


"""
    spectral_calibration(lasers::WeightedArray{T,2}, lamp_spectra::Vector, profiles::AbstractVector; calib_params::FastPICParams=FastPICParams()) where {T}

Perform complete spectral wavelength calibration using laser and lamp data.

This is the main entry point for spectral calibration, combining:
1. Laser line fitting for initial wavelength solutions
2. Template estimation from lamp spectra
3. Iterative refinement of wavelength calibrations

# Arguments
- `lasers::WeightedArray{T,2}`: 2D laser calibration data
- `lamp_spectra::Vector`: Extracted lamp spectra for each lenslet
- `profiles::AbstractVector`: Spatial profile models for each lenslet
- `calib_params::FastPICParams`: Configuration parameters

# Returns
- `Tuple{Vector, Vector{Float64}, Vector{Float64}, AbstractVector, Vector{Bool}}`: 
  - Wavelength calibration coefficients
  - Common lamp template spectrum
  - Transmission factors for each lenslet
  - Common wavelength grid
  - Updated valid lenslets mask

# Examples
```julia
coefs, template, transmission, λ_grid, valid = spectral_calibration(
    laser_data, lamp_spectra, profiles; calib_params=my_params
)
```
"""
function spectral_calibration(
        lasers::WeightedArray{T, 2},
        lamp_spectra::Vector{L},
        profiles::AbstractVector{<:Union{Nothing, Profile}};
        calib_params::FastPICParams = FastPICParams()
    ) where {T, L <: Union{Nothing, WeightedArray{T, 1}}}

    @unpack_FastPICParams calib_params

    laser_spectra, las, coefs, λs = laser_calibration!(L, lasers, profiles; calib_params = calib_params)
    lλ = build_λrange(λs; superres = spectral_superres)

    coefs, template, transmission = recalibrate_wavelengths(
        profiles,
        lλ,
        coefs,
        spectral_final_order,
        lamp_spectra,
        laser_spectra,
        lasers_λs,
        las;
        regul = spectral_recalibration_regul,
        loop = spectral_recalibration_loop,
        ntasks = ntasks,
        verbose = spectral_calibration_verbose
    )
    return coefs, template, transmission, lλ
end


"""
    laser_calibration!(::Type{L}, lasers, profiles; calib_params::FastPICParams=FastPICParams()) where {L}

Extract and fit laser spectra to establish initial wavelength calibrations.

# Arguments
- `::Type{L}`: Type for laser spectrum storage
- `lasers`: 2D laser calibration data
- `profiles`: Spatial profile models for spectrum extraction
- `calib_params::FastPICParams`: Configuration parameters

# Returns
- `Tuple{Vector, Vector, Vector, Vector, Vector{Bool}}`:
  - Extracted laser spectra
  - Fitted laser models  
  - Initial wavelength coefficients
  - Initial wavelength grids
  - Updated valid lenslets mask

# Side Effects
Updates the `profiles` vector in-place, setting invalid entries to `nothing`.
"""
function laser_calibration!(
        ::Type{L},
        lasers,
        profiles;
        calib_params::FastPICParams = FastPICParams()
    ) where {L}
    @unpack_FastPICParams calib_params

    laser_spectra = Vector{L}(undef, NLENS)
    laser_model = LaserModel([7.0, 20.0, 35.0], [2.0, 2.0, 2.0])
    coefs = Vector{Union{Nothing, Vector{Float64}}}(undef, NLENS)
    λs = Vector{Union{Nothing, Vector{Float64}}}(undef, NLENS)
    las = Vector{typeof(laser_model)}(undef, NLENS)

    fill!(coefs, nothing)
    fill!(λs, nothing)
    fill!(laser_spectra, nothing)
    valid_lenslets = map(!isnothing, profiles)
    progressbar = spectral_calibration_verbose ? Progress(sum(valid_lenslets); showspeed = true, desc = "Spectral calibration") : nothing

    tforeach(findall(valid_lenslets); ntasks = ntasks) do i
        if sum(view(lasers, profiles[i].bbox).precision) != 0
            try
                laser_spectra[i] = extract_spectrum(lasers, profiles[i]; restrict = laser_extract_restrict)
                las[i] = fit_laser(laser_spectra[i], laser_model)

                if std(las[i].position .- laser_model.position) > 1
                    throw("Laser position too far from initial guess for lenslet $i")
                end
                W = get_laser_precision(las[i], laser_spectra[i])
                if any(diag(W) .< 1.0e-4)
                    throw("W singular  for lenslet $i")
                end
                coefs[i] = laser_calibration(
                    spectral_initial_order, profiles[i].ycenter - profiles[i].bbox.ymin, lasers_λs, las[i].position, W
                )
                if any(isnan.(coefs[i]))
                    throw("NaN found in coefs for lenslet $i")
                end
                λs[i] = get_wavelength(coefs[i], profiles[i].ycenter - profiles[i].bbox.ymin, axes(laser_spectra[i], 1))

            catch e
                @debug "Error on lenslet $i" exception = (e, catch_backtrace())
                profiles[i] = nothing
            end
        end
        spectral_calibration_verbose && next!(progressbar)
    end
    spectral_calibration_verbose && finish!(progressbar)
    return laser_spectra, las, coefs, λs
end
