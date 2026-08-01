
include("simulations_functions.jl")

using Random
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using RCall
using FillArrays # Required for mvnormcdf
using MvNormalCDF
using ProgressMeter
use_accelerated_blas!()


sim_reps = Int(simcfg("simulation_comparison", "sim_reps", 100))
sim_reps_big = Int(simcfg("simulation_comparison", "sim_reps_big", 100))

using Distributions









# Run pnorm qmc_pnorm



T = Float64
n_dim = 2^12
max_pts = 2^11 * 10
seed = 42


independent_variance = 1.0
common_variance = 1.0
Σ = fixed_dense(n_dim; jitter=independent_variance)
k = (quantile(Normal(), (1 + 0.25^(1 / n_dim)) / 2))
a = -sqrt.(diag(Σ)) * k
b = (sqrt.(diag(Σ))) * k

opts = QMC_opts(Float64;
    chol_block_size=2^9, chol_block_size2=2^7, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^6)



# opts_s = QMC_opts(T; m=max_pts, max_pts=max_pts, chol_block_size=2^8, chol_block_size2=2^9, max_abs_err=0.0, block_size_i=2^6, block_size_i2=2^6, block_size_j=2^6)



# @time data_s = QMCDataSparse(Σ, a, b, opts_s, MersenneTwister(seed), :Richtmyer);


@time data = QMCData(Σ, a, b, opts, MersenneTwister(seed), :Richtmyer);


# @time vs, es, n_pts_s = qmc_pnorm!(data_s);



@time val, err, n_pts = qmc_pnorm!(data);




"""
    equicorrelated_mvn_probability(α, β, lower, upper; rtol=1e-12)

Compute `P(lower ≤ X ≤ upper)` for
`X ~ N(0, αI + β * 11')`, where `α > 0` and `β ≥ 0`.

The representation `Xᵢ = sqrt(α) * εᵢ + sqrt(β) * Z` reduces the
multivariate probability to a one-dimensional expectation over `Z ~ N(0, 1)`.
Conditional interval probabilities are accumulated on the log scale.
"""
function equicorrelated_mvn_probability(
    α::Real,
    β::Real,
    lower::AbstractVector,
    upper::AbstractVector;
    rtol::Real=1e-12,
)
    length(lower) == length(upper) ||
        throw(DimensionMismatch("lower and upper must have the same length"))
    all(lower .<= upper) ||
        throw(ArgumentError("each lower bound must not exceed its upper bound"))
    isfinite(α) && α > 0 ||
        throw(ArgumentError("α must be finite and positive"))
    isfinite(β) && β >= 0 ||
        throw(ArgumentError("β must be finite and nonnegative"))

    standard_normal = Normal()
    inv_sqrt_α = inv(sqrt(float(α)))
    sqrt_β = sqrt(float(β))

    return Distributions.expectation(standard_normal; rtol=rtol) do z
        common_shift = sqrt_β * z
        log_conditional_probability = 0.0

        @inbounds for i in eachindex(lower, upper)
            standardized_upper = (upper[i] - common_shift) * inv_sqrt_α
            standardized_lower = (lower[i] - common_shift) * inv_sqrt_α
            log_conditional_probability += logdiffcdf(
                standard_normal,
                standardized_upper,
                standardized_lower,
            )
        end

        return exp(log_conditional_probability)
    end
end

@time val_actual = equicorrelated_mvn_probability(
    independent_variance,
    common_variance,
    a,
    b,
)

@show abs(val - val_actual)
