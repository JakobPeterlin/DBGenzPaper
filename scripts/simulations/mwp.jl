
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




n_dim = 2^4
max_pts = 2^11 * 10
seed = 1


Σ = rand_spd(n_dim)
k = quantile(Normal(), (1 + 0.25^(1 / n_dim)) / 2)
a = -sqrt.(diag(Σ)) * k
b = sqrt.(diag(Σ)) * k



<<<<<<< HEAD
opts = QMC_opts(Float64; m=max_pts, max_pts=max_pts, chol_block_size=2^9, chol_block_size2=2^9, max_abs_err=0.0, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^6)
=======
opts = QMC_opts(Float64; chol_block_size=2^6, m=2^max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^10, block_size_i2=2^6, block_size_j=2^7)
>>>>>>> cd0725f (Simulations etc. on M2U.)


#@time data = QMCData(Σ, a, b, opts, MersenneTwister(seed), :Richtmyer);

#@time val, err, _ = qmc_pnorm!(data);

@time val, err, t = qmc_pnorm!(QMCData(Σ, a, b, opts, MersenneTwister(seed), :Richtmyer););
