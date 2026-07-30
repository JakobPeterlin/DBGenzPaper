
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
n_dim = 2^8
max_pts = 2^11 * 10
seed = 42


Σ = (mattern_cov2(n_dim))
Σ = (rand_spd(n_dim))
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