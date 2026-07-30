include("simulations_functions.jl")

using Random
using Statistics
using LinearAlgebra
using DataFrames
using Distributions

use_accelerated_blas!()


# REPL-only diagnostic: keep all results in memory and return one two-row
# DataFrame. Do not write CSV files or any other simulation artifacts.
# Match the mattern_cov2, n = 256, m = 245760 case from simulation_sparse
# with the paper.toml settings. The paper's total point budget is split over
# the 12 randomized QMC replications used by QMC_opts.
T = Float64
n_dim = 2^8
max_pts = 2^11 * 120
n_qmc_reps = 12
pts_per_qmc_rep = cld(max_pts, n_qmc_reps)
n_reps = 100
seed = 42

Σ = mattern_cov2(n_dim)
k = quantile(Normal(), (1 + 0.25^(1 / n_dim)) / 2)
a = -k .* sqrt.(diag(Σ))
b = k .* sqrt.(diag(Σ))


dense_opts = use_MKL_instead_of_ACC ?
             QMC_opts(T;
    chol_block_size=2^9,
    chol_block_size2=2^9,
    m=pts_per_qmc_rep,
    max_pts=pts_per_qmc_rep,
    max_abs_err=0.0,
    block_size_i=2^9,
    block_size_i2=2^6,
    block_size_j=2^6,
    n_reps=n_qmc_reps,
) :
             QMC_opts(T;
    chol_block_size=2^5,
    chol_block_size2=2^7,
    m=pts_per_qmc_rep,
    max_pts=pts_per_qmc_rep,
    max_abs_err=0.0,
    block_size_i=2^10,
    block_size_i2=2^6,
    block_size_j=2^8,
    n_reps=n_qmc_reps,
)

nnz_mul = count(abs.(Σ .> eps(T))) / length(Σ) < 0.1 ? 4 : 1
sparse_opts = use_MKL_instead_of_ACC ?
              QMC_opts(T;
    chol_block_size=2^8,
    chol_block_size2=2^9,
    m=pts_per_qmc_rep,
    max_pts=pts_per_qmc_rep,
    max_abs_err=0.0,
    block_size_i=2^6,
    block_size_i2=2^6,
    block_size_j=2^6,
    n_reps=n_qmc_reps,
) :
              QMC_opts(T;
    chol_block_size=2^5,
    chol_block_size2=2^7,
    m=pts_per_qmc_rep,
    max_pts=pts_per_qmc_rep,
    max_abs_err=0.0,
    block_size_i=2^6,
    block_size_i2=2^6,
    block_size_j=2^6 * nnz_mul,
    n_reps=n_qmc_reps,
)


function run_method(constructor, method, opts, Σ, a, b, n_reps, seed)
    rows = DataFrame(
        method=String[],
        rep=Int[],
        value=Float64[],
        est_error=Float64[],
        time=Float64[],
        pts_per_qmc_rep=Int[],
    )

    for rep in 1:n_reps
        result = nothing
        elapsed = @elapsed result = qmc_pnorm!(
            constructor(Σ, a, b, opts, MersenneTwister(seed + rep), :Richtmyer),
        )
        value, est_error, used_pts = result
        push!(rows, (
            method=method,
            rep=rep,
            value=value,
            est_error=est_error,
            time=elapsed,
            pts_per_qmc_rep=used_pts,
        ))
    end

    return rows
end


raw_results = vcat(
    run_method(QMCData, "QMCData", dense_opts, Σ, a, b, n_reps, seed),
    run_method(QMCDataSparse, "QMCDataSparse", sparse_opts, Σ, a, b, n_reps, seed),
)

comparison_df = combine(groupby(raw_results, :method),
    :value => median => :median,
    :value => std => :sd,
    :value => minimum => :min,
    :value => maximum => :max,
    :value => (x -> quantile(x, 0.25)) => :Q1,
    :value => (x -> quantile(x, 0.75)) => :Q3,
    :est_error => mean => :mean_est_error,
    :time => median => :median_time,
    :time => minimum => :min_time,
    :pts_per_qmc_rep => (only ∘ unique) => :pts_per_qmc_rep,
)

comparison_df[!, :n_dim] .= n_dim
comparison_df[!, :max_pts] .= max_pts
comparison_df[!, :n_qmc_reps] .= n_qmc_reps
comparison_df[!, :n_reps] .= n_reps
comparison_df[!, :first_seed] .= seed + 1
comparison_df[!, :last_seed] .= seed + n_reps

comparison_df
