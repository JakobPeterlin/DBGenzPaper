
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
n_dim = 2^10
max_pts = 2^11 * 10
seed = 42


Σ = (mattern_cov1(n_dim))
k = (quantile(Normal(), (1 + 0.25^(1 / n_dim)) / 2))
a = -sqrt.(diag(Σ)) * k
b = (sqrt.(diag(Σ))) * k

Σ2 = copy(Σ)






opts = QMC_opts(Float64;
    chol_block_size=2^9, chol_block_size2=2^7, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^6)



opts_s = QMC_opts(T; m=max_pts, max_pts=max_pts, chol_block_size=2^8, chol_block_size2=2^9, max_abs_err=0.0, block_size_i=2^6, block_size_i2=2^6, block_size_j=2^6)



@time data_s = QMCDataSparse(Σ, a, b, opts_s, MersenneTwister(seed), :Richtmyer);


@time data = QMCData(Σ, a, b, opts, MersenneTwister(seed), :Richtmyer);


@time vs, es, t = qmc_pnorm!(QMCDataSparse(Σ2, a, b, opts_s, MersenneTwister(seed), :Richtmyer););



@time val, err, t = qmc_pnorm!(QMCData(Σ, a, b, opts_s, MersenneTwister(seed), :Richtmyer););




val - vs


##

n_reps = 100
opts_r11 = QMC_opts(Float32; m=2^11, max_pts=2^11, max_abs_err=0.0)
opts_r11x10 = QMC_opts(Float32; m=2^11 * 10, max_pts=2^11 * 10, max_abs_err=0.0)

@time r11 = [qmc_pnorm!(QMCData(copy(Σ), copy(a), copy(b), opts_r11, MersenneTwister(seed + i), :Richtmyer))[1] for i in 1:n_reps]
@time r11x10 = [qmc_pnorm!(QMCData(copy(Σ), copy(a), copy(b), opts_r11x10, MersenneTwister(seed + i), :Richtmyer))[1] for i in 1:n_reps]
@time s11 = [qmc_pnorm!(QMCData(copy(Σ), copy(a), copy(b), opts_s11, MersenneTwister(seed + i), :Sobol))[1] for i in 1:n_reps]
@time s11x10 = [qmc_pnorm!(QMCData(copy(Σ), copy(a), copy(b), opts_s11x10, MersenneTwister(seed + i), :Sobol))[1] for i in 1:n_reps]

e = Float64.(sd.((r11, r11x10, s11, s11x10)))
