include("simulations_functions.jl")

using Random
using Statistics
using LinearAlgebra
using DataFrames
using Distributions
using RCall
using CSV



function run_pnorm(Σ, a, b; max_pts::Int=2^10, seed=0)
    rng = MersenneTwister(seed)

    opts = use_MKL_instead_of_ACC ? QMC_opts(Float64;
        chol_block_size=2^9, chol_block_size2=2^9, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^6) :
           QMC_opts(Float64; chol_block_size=2^5, chol_block_size2=2^7, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^10, block_size_i2=2^6, block_size_j=2^8)


    t = @elapsed (val, err, _) = qmc_pnorm!(QMCData((Σ), (a), (b), opts, rng, :Richtmyer))

    return val, err, t
end





function run_pnorm32(Σ, a, b; max_pts=2^10, seed=0)
    rng = MersenneTwister(seed)
    opts = use_MKL_instead_of_ACC ? QMC_opts(Float32;
        chol_block_size=2^9, chol_block_size2=2^9, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^6) :
           QMC_opts(Float32; chol_block_size=2^6, chol_block_size2=2^8, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^10, block_size_i2=2^6, block_size_j=2^8)


    t = @elapsed (val, err, _) = qmc_pnorm!(QMCData(Σ, a, b, opts, rng, :Richtmyer))
    return val, err, t
end





function run_pnorm_sparse(Σ, a, b; max_pts=2^10, seed=0)
    nnz_mul = count(abs.(Σ .> eps(Float64))) / prod(size(Σ)) < 0.1 ? 4 : 1

    rng = MersenneTwister(seed)
    opts = use_MKL_instead_of_ACC ? QMC_opts(Float64;
        chol_block_size=2^8, chol_block_size2=2^9, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^6, block_size_i2=2^6, block_size_j=2^6) :
           QMC_opts(Float64; chol_block_size=2^5, chol_block_size2=2^7, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^6, block_size_i2=2^6, block_size_j=2^6 * nnz_mul)

    t = @elapsed (val, err, _) = qmc_pnorm!(QMCDataSparse(Σ, a, b, opts, rng, :Richtmyer))
    return val, err, t
end







function run_pnorm_sparse32(Σ, a, b; max_pts=2^10, seed=0)
    nnz_mul = count(abs.(Σ .> eps(Float64))) / prod(size(Σ)) < 0.1 ? 4 : 1
    rng = MersenneTwister(seed)
    opts = use_MKL_instead_of_ACC ? QMC_opts(Float32;
        chol_block_size=2^8, chol_block_size2=2^9, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^6, block_size_i2=2^6, block_size_j=2^6) :
           QMC_opts(Float32; chol_block_size=2^5, chol_block_size2=2^7, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^6, block_size_i2=2^6, block_size_j=2^6 * nnz_mul)

    t = @elapsed (val, err, _) = qmc_pnorm!(QMCDataSparse(Σ, a, b, opts, rng, :Richtmyer))
    return val, err, t
end





function run_tlrmvnmvt(a, b, Σ, maxpts)
    @rput a b Σ maxpts
    R"""
    t0 <- proc.time()
    res <- tlrmvnmvt::pmvn(
      lower = a, upper = b, mu = rep(0, length(a)), sigma = Σ,
      algorithm = tlrmvnmvt::TLRQMC(N = maxpts)
     )
    t_elapsed <- (proc.time() - t0)[["elapsed"]]
    """
    val = rcopy(R"res[1]")
    err = rcopy(R"attr(res, 'error')")
    t = rcopy(R"t_elapsed")
    return Float64(val), Float64(err), Float64(t)
end


function compare_sparse_methods(; n_dim=1000, n_reps=20, seed=42, qmc_pts=9600)
    Random.seed!(seed)

    has_tlr = rcopy(R"requireNamespace('tlrmvnmvt', quietly=TRUE)")
    has_tlr && R"library(tlrmvnmvt)"

    M = mattern_cov1(n_dim)
    k = quantile(Normal(), (1 + 0.25^(1 / n_dim)) / 2)
    a = -k * sqrt.(diag(M))
    b = k * sqrt.(diag(M))

    results = DataFrame(
        rep=Int[],
        method=String[],
        value=Float64[],
        time=Float64[],
        error=Union{Float64,Missing}[],
        est_error=Float64[]
    )

    for rep in 1:n_reps
        val, err, t = run_pnorm(M, a, b; max_pts=ceil(Int, qmc_pts / 12), seed=seed + rep)
        push!(results, (rep=rep, method="pnorm", value=val, time=t, error=missing, est_error=err))
    end

    for rep in 1:n_reps
        val, err, t = run_pnorm32(M, a, b; max_pts=ceil(Int, qmc_pts / 12), seed=seed + rep)
        push!(results, (rep=rep, method="pnorm32", value=val, time=t, error=missing, est_error=err))
    end

    for rep in 1:n_reps
        val, err, t = run_pnorm_sparse(M, a, b; max_pts=ceil(Int, qmc_pts / 12), seed=seed + rep)
        push!(results, (rep=rep, method="pnorm_sparse", value=val, time=t, error=missing, est_error=err))
    end


    for rep in 1:n_reps
        val, err, t = run_pnorm_sparse32(M, a, b; max_pts=ceil(Int, qmc_pts / 12), seed=seed + rep)
        push!(results, (rep=rep, method="pnorm_sparse32", value=val, time=t, error=missing, est_error=err))
    end

    println("running tlr")

    if has_tlr
        for rep in 1:n_reps
            println("rep $rep out of $n_reps")
            val, err, t = run_tlrmvnmvt(a, b, M, Int(ceil(qmc_pts / 20)))
            push!(results, (rep=rep, method="tlr", value=val, time=t, error=missing, est_error=err))
        end
    end

    baseline_method = has_tlr ? "tlr" : "pnorm"
    refs = Dict(row.rep => row.value for row in eachrow(results[results.method.==baseline_method, :]))
    for i in 1:nrow(results)
        if haskey(refs, results.rep[i])
            results.error[i] = abs(results.value[i] - refs[results.rep[i]])
        end
    end

    return results, baseline_method
end


function compare_dense_methods(; n_dim=100, n_reps=20, seed=42, qmc_pts=9600, mat_fun=mattern_cov2)
    Random.seed!(seed)

    has_tlr = rcopy(R"requireNamespace('tlrmvnmvt', quietly=TRUE)")
    has_tlr && R"library(tlrmvnmvt)"

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
        est_error=Float64[]
    )

    for rep in 1:n_reps
        val, err, t = run_pnorm(M, a, b; max_pts=ceil(Int, qmc_pts / 12), seed=seed + rep)
        push!(results, (rep=rep, method="pnorm", value=val, time=t, error=missing, est_error=err))
    end

    for rep in 1:n_reps
        val, err, t = run_pnorm32(M, a, b; max_pts=ceil(Int, qmc_pts / 12), seed=seed + rep)
        push!(results, (rep=rep, method="pnorm32", value=val, time=t, error=missing, est_error=err))
    end

    for rep in 1:n_reps
        val, err, t = run_pnorm_sparse(M, a, b; max_pts=ceil(Int, qmc_pts / 12), seed=seed + rep)
        push!(results, (rep=rep, method="pnorm_sparse", value=val, time=t, error=missing, est_error=err))
    end

    for rep in 1:n_reps
        val, err, t = run_pnorm_sparse32(M, a, b; max_pts=ceil(Int, qmc_pts / 12), seed=seed + rep)
        push!(results, (rep=rep, method="pnorm_sparse32", value=val, time=t, error=missing, est_error=err))
    end
    println("running tlr")

    if has_tlr
        for rep in 1:n_reps
            println("rep $rep out of $n_reps")

            val, err, t = run_tlrmvnmvt(a, b, M, Int(ceil(qmc_pts / 20)))
            push!(results, (rep=rep, method="tlr", value=val, time=t, error=missing, est_error=err))
        end
    end

    baseline_method = has_tlr ? "tlr" : "pnorm"
    refs = Dict(row.rep => row.value for row in eachrow(results[results.method.==baseline_method, :]))
    for i in 1:nrow(results)
        if haskey(refs, results.rep[i])
            results.error[i] = abs(results.value[i] - refs[results.rep[i]])
        end
    end

    return results, baseline_method
end


function summarize_results(df::DataFrame)
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
        :est_error => mean => :se_est
    )

    time_summary = combine(groupby(df, :method),
        :time => median => :median,
        :time => minimum => :min,
        :time => maximum => :max,
        :time => (x -> quantile(x, 0.25)) => :Q1,
        :time => (x -> quantile(x, 0.75)) => :Q3
    )

    return val_summary, time_summary
end


function run_all_simulations_sparse(ns, n_ptss, n_reps)
    val_df = DataFrame()
    t_df = DataFrame()

    @showprogress for n in ns
        for n_pts in n_ptss
            results, baseline = compare_sparse_methods(n_dim=n, qmc_pts=n_pts, n_reps=n_reps)
            val_df_i, t_df_i = summarize_results(results)

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


function run_all_simulations_sparse2(ns, n_ptss, n_reps)
    val_df = DataFrame()
    t_df = DataFrame()

    @showprogress for n in ns
        for n_pts in n_ptss
            results, baseline = compare_dense_methods(n_dim=n, qmc_pts=n_pts, n_reps=n_reps)
            val_df_i, t_df_i = summarize_results(results)

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


function run_all_simulations_dense(ns, n_ptss, n_reps)
    val_df = DataFrame()
    t_df = DataFrame()

    @showprogress for n in ns
        for n_pts in n_ptss
            results, baseline = compare_dense_methods(n_dim=n, qmc_pts=n_pts, n_reps=n_reps, mat_fun=fixed_dense)
            val_df_i, t_df_i = summarize_results(results)

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


ns = 2 .^ [Int(x) for x in simcfg("simulation_sparse", "n_powers", [8, 9, 10, 11, 12])]
n_ptss = [Int(x) for x in simcfg("simulation_sparse", "m_values", [2^11 * 120])]
n_reps = Int(simcfg("simulation_sparse", "n_reps", 100))

sparse_vals, sparse_times = run_all_simulations_sparse(ns, n_ptss, n_reps)
sparse_vals[!, :matrix] .= "mattern_cov1"
sparse_times[!, :matrix] .= "mattern_cov1"

dense_vals, dense_times = run_all_simulations_sparse2(ns, n_ptss, n_reps)
dense_vals[!, :matrix] .= "mattern_cov2"
dense_times[!, :matrix] .= "mattern_cov2"

fixed_dense_vals, fixed_dense_times = run_all_simulations_dense(ns, n_ptss, n_reps)
fixed_dense_vals[!, :matrix] .= "fixed_dense"
fixed_dense_times[!, :matrix] .= "fixed_dense"

combined_vals = vcat(sparse_vals, dense_vals, fixed_dense_vals; cols=:union)
combined_times = vcat(sparse_times, dense_times, fixed_dense_times; cols=:union)

CSV.write(sim_resultpath("sparse_dense_vals.csv"), combined_vals)
CSV.write(sim_resultpath("sparse_dense_times.csv"), combined_times)