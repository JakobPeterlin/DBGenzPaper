include("simulations_functions.jl")

using Random
using Statistics
using LinearAlgebra
using DataFrames
using Distributions
using CSV


function run_pnorm32_only(Σ, a, b; max_pts=2^10, seed=0)
    rng = MersenneTwister(seed)
    opts = use_MKL_instead_of_ACC ? QMC_opts(Float32;
        chol_block_size=2^9, chol_block_size2=2^9, m=max_pts, max_pts=max_pts,
        max_abs_err=0.0, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^6) :
           QMC_opts(Float32;
               chol_block_size=2^6, chol_block_size2=2^8, m=max_pts,
               max_pts=max_pts, max_abs_err=0.0, block_size_i=2^10,
               block_size_i2=2^6, block_size_j=2^8)

    t = @elapsed (val, err, _) = qmc_pnorm!(
        QMCData(Σ, a, b, opts, rng, :Richtmyer)
    )
    return val, err, t
end


function run_pnorm_sparse32_only(Σ, a, b; max_pts=2^10, seed=0)
    nnz_mul = count(abs.(Σ .> eps(Float64))) / prod(size(Σ)) < 0.1 ? 4 : 1
    rng = MersenneTwister(seed)
    opts = use_MKL_instead_of_ACC ? QMC_opts(Float32;
        chol_block_size=2^8, chol_block_size2=2^9, m=max_pts, max_pts=max_pts,
        max_abs_err=0.0, block_size_i=2^6, block_size_i2=2^6, block_size_j=2^6) :
           QMC_opts(Float32;
               chol_block_size=2^5, chol_block_size2=2^7, m=max_pts,
               max_pts=max_pts, max_abs_err=0.0, block_size_i=2^6,
               block_size_i2=2^6, block_size_j=2^6 * nnz_mul)

    t = @elapsed (val, err, _) = qmc_pnorm!(
        QMCDataSparse(Σ, a, b, opts, rng, :Richtmyer)
    )
    return val, err, t
end


function compare_fp32_methods(;
    n_dim=1000,
    n_reps=20,
    seed=42,
    qmc_pts=9600,
    mat_fun=mattern_cov1,
)
    Random.seed!(seed)

    M = mat_fun(n_dim)
    k = quantile(Normal(), (1 + 0.25^(1 / n_dim)) / 2)
    a = -k * sqrt.(diag(M))
    b = k * sqrt.(diag(M))

    results = DataFrame(
        rep=Int[],
        method=String[],
        value=Float64[],
        time=Float64[],
        error=Union{Float64,Missing}[],
        est_error=Float64[],
    )

    for rep in 1:n_reps
        val, err, t = run_pnorm32_only(
            M,
            a,
            b;
            max_pts=ceil(Int, qmc_pts / 12),
            seed=seed + rep,
        )
        push!(results, (
            rep=rep,
            method="pnorm32",
            value=val,
            time=t,
            error=missing,
            est_error=err,
        ))
    end

    for rep in 1:n_reps
        val, err, t = run_pnorm_sparse32_only(
            M,
            a,
            b;
            max_pts=ceil(Int, qmc_pts / 12),
            seed=seed + rep,
        )
        push!(results, (
            rep=rep,
            method="pnorm_sparse32",
            value=val,
            time=t,
            error=missing,
            est_error=err,
        ))
    end

    return results
end


function summarize_fp32_results(df::DataFrame)
    if isempty(df)
        return DataFrame(), DataFrame()
    end

    val_summary = combine(groupby(df, :method),
        :value => median => :median,
        :value => std => :sd,
        :value => minimum => :min,
        :value => maximum => :max,
        :value => (x -> quantile(x, 0.25)) => :Q1,
        :value => (x -> quantile(x, 0.75)) => :Q3,
        :est_error => mean => :se_est,
    )

    time_summary = combine(groupby(df, :method),
        :time => median => :median,
        :time => minimum => :min,
        :time => maximum => :max,
        :time => (x -> quantile(x, 0.25)) => :Q1,
        :time => (x -> quantile(x, 0.75)) => :Q3,
    )

    return val_summary, time_summary
end


function run_all_fp32_simulations(ns, n_ptss, n_reps; mat_fun)
    val_df = DataFrame()
    t_df = DataFrame()

    @showprogress for n in ns
        for n_pts in n_ptss
            results = compare_fp32_methods(
                n_dim=n,
                qmc_pts=n_pts,
                n_reps=n_reps,
                mat_fun=mat_fun,
            )
            val_df_i, t_df_i = summarize_fp32_results(results)

            val_df_i[!, :n] = fill(n, nrow(val_df_i))
            val_df_i[!, :n_pts] = fill(n_pts, nrow(val_df_i))
            t_df_i[!, :n] = fill(n, nrow(t_df_i))
            t_df_i[!, :n_pts] = fill(n_pts, nrow(t_df_i))

            append!(val_df, val_df_i)
            append!(t_df, t_df_i)
        end
    end

    return val_df, t_df
end


ns = 2 .^ [Int(x) for x in simcfg(
    "simulation_FP32",
    "n_powers",
    [8, 9, 10, 11, 12],
)]
n_ptss = [Int(x) for x in simcfg(
    "simulation_FP32",
    "m_values",
    [2^11 * 120],
)]
n_reps = Int(simcfg("simulation_FP32", "n_reps", 100))

fp32_sparse_vals, fp32_sparse_times = run_all_fp32_simulations(
    ns,
    n_ptss,
    n_reps;
    mat_fun=mattern_cov1,
)
fp32_sparse_vals[!, :matrix] .= "mattern_cov1"
fp32_sparse_times[!, :matrix] .= "mattern_cov1"

fp32_dense_vals, fp32_dense_times = run_all_fp32_simulations(
    ns,
    n_ptss,
    n_reps;
    mat_fun=mattern_cov2,
)
fp32_dense_vals[!, :matrix] .= "mattern_cov2"
fp32_dense_times[!, :matrix] .= "mattern_cov2"

fp32_fixed_dense_vals, fp32_fixed_dense_times = run_all_fp32_simulations(
    ns,
    n_ptss,
    n_reps;
    mat_fun=fixed_dense,
)
fp32_fixed_dense_vals[!, :matrix] .= "fixed_dense"
fp32_fixed_dense_times[!, :matrix] .= "fixed_dense"

combined_fp32_vals = vcat(
    fp32_sparse_vals,
    fp32_dense_vals,
    fp32_fixed_dense_vals;
    cols=:union,
)
combined_fp32_times = vcat(
    fp32_sparse_times,
    fp32_dense_times,
    fp32_fixed_dense_times;
    cols=:union,
)

CSV.write(sim_resultpath("sparse_dense_vals_FP32.csv"), combined_fp32_vals)
CSV.write(sim_resultpath("sparse_dense_times_FP32.csv"), combined_fp32_times)
