
include(joinpath(@__DIR__, "setup.jl"))
using DBGenzPaper

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
function run_pnorm(Σ, a, b; max_pts::Int=2^10, seed=0)
    rng = MersenneTwister(seed)
    n = length(a)
    opts = use_MKL_instead_of_ACC ? QMC_opts(Float64; chol_block_size=2^6, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^6) :
           QMC_opts(Float64; chol_block_size=2^6, m=2^11, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^10, block_size_i2=2^6, block_size_j=2^7)


    t = @elapsed (val, err, _) = qmc_pnorm!(QMCData((Σ), (a), (b), opts, rng, :Richtmyer))

    return val, err, t
end






# Run mvnormcdf
function run_mvnormcdf(Σ, a, b; m=10^5, seed=0)
    rng = MersenneTwister(seed)
    res = (0.0, 0.0)
    t = @elapsed res = mvnormcdf(Σ, a, b; m=m, rng=rng)
    return res[1], res[2], t
end



# Run mvtnorm::pmvnorm
function run_mvtnorm(a, b, Σ; abseps=1e-12, maxpts=10)
    @rput a b Σ abseps maxpts
    R"""
    t0 <- proc.time()
    res <- mvtnorm::pmvnorm(
      lower = a, upper = b, sigma = Σ,
      algorithm = mvtnorm::GenzBretz(maxpts = maxpts, abseps = abseps)
    )
    t_elapsed <- (proc.time() - t0)[["elapsed"]]
    """
    val = rcopy(R"res[1]")
    err = rcopy(R"attr(res, 'error')")
    t = rcopy(R"t_elapsed")
    return Float64(val), Float64(err), Float64(t)
end

# Run tlrmvnmvt::pmvn
function run_tlrmvnmvt(a, b, Σ, maxpts)
    @rput a b Σ maxpts
    R"""
    t0 <- proc.time()
    res <- tlrmvnmvt::pmvn(
      lower = a, upper = b, mu = rep(0, length(a)), sigma = Σ,
      algorithm = tlrmvnmvt::GenzBretz(N = maxpts)
     )
    t_elapsed <- (proc.time() - t0)[["elapsed"]]
    """
    val = rcopy(R"res[1]")
    err = rcopy(R"attr(res, 'error')")
    t = rcopy(R"t_elapsed")
    return Float64(val), Float64(err), Float64(t)
end













function run_simulation(;
    n_dim=100,
    n_reps=10^3,
    seed=42,
    qmc_pts=9600)
    Random.seed!(seed)

    # Check which R packages are available
    has_mvtnorm = rcopy(R"requireNamespace('mvtnorm', quietly=TRUE)")
    has_tlr = rcopy(R"requireNamespace('tlrmvnmvt', quietly=TRUE)")
    has_mvtnorm && R"library(mvtnorm)"
    has_tlr && R"library(tlrmvnmvt)"
    R"set.seed(42)"

    # Pre-generate test case to ensure consistency


    Σ = random_spd(n_dim)
    k = quantile(Normal(), (1 + 0.25^(1 / n_dim)) / 2)
    a = -sqrt.(diag(Σ)) * k
    b = sqrt.(diag(Σ)) * k


    # DataFrame to store results
    results = DataFrame(
        rep=Int[],
        method=String[],
        value=Float64[],
        time=Float64[],
        error=Union{Float64,Missing}[],
        est_error=Float64[]
    )

    baseline_method = "none"

    # 1. pnorm
    for rep in 1:n_reps
        val, err, t = run_pnorm(Σ, a, b;
            max_pts=Int(qmc_pts / 12), seed=seed + rep)
        push!(results, (rep=rep, method="pnorm", value=val, time=t, error=missing, est_error=err))
    end

    # 2. mvnormcdf
    for rep in 1:n_reps
        val, err, t = run_mvnormcdf(Σ, a, b; m=qmc_pts, seed=seed + rep)
        push!(results, (rep=rep, method="mvnormcdf", value=val, time=t, error=missing, est_error=err))
    end


    # 3. mvnmvt (was mvtnorm)
    if has_mvtnorm
        for rep in 1:n_reps

            val, err, t = run_mvtnorm(a, b, Σ; maxpts=qmc_pts)
            push!(results, (rep=rep, method="mvnmvt", value=val, time=t, error=missing, est_error=err))
        end

    end

    # 4. tlr (was tlrmvnmvt)
    if has_tlr
        for rep in 1:n_reps

            val, err, t = run_tlrmvnmvt(a, b, Σ, Int(ceil(qmc_pts / 20)))
            push!(results, (rep=rep, method="tlr", value=val, time=t, error=missing, est_error=err))

        end

    end

    # Determine reference values for error calculation
    if has_mvtnorm && any(results.method .== "mvnmvt")
        baseline_method = "mvnmvt"
    elseif any(results.method .== "pnorm")
        baseline_method = "pnorm"
    elseif !isempty(results)
        baseline_method = results.method[1]
    end

    if baseline_method != "none"
        # Create a lookup for reference values
        refs = Dict(row.rep => row.value for row in eachrow(results[results.method.==baseline_method, :]))
        for i in 1:nrow(results)
            if haskey(refs, results.rep[i])
                results.error[i] = abs(results.value[i] - refs[results.rep[i]])
            end
        end
    end

    return results, baseline_method
end

"""
    run_simulation_big(; n_dim=100, n_reps=10^3, seed=42, qmc_pts=9600)

Like `run_simulation`, but runs the Julia methods and (optionally) the `tlrmvnmvt` R method:
- `pnorm` (QMC via `QMCData`/`qmc_pnorm!`)
- `mvnormcdf` (via `MvNormalCDF`)
- `tlr` (via `tlrmvnmvt::pmvn`, only if the R package is available)
"""
function run_simulation_big(;
    n_dim=100,
    n_reps=10^3,
    seed=42,
    qmc_pts=9600)
    Random.seed!(seed)

    # Optional R method (tlrmvnmvt)
    has_tlr = rcopy(R"requireNamespace('tlrmvnmvt', quietly=TRUE)")
    has_tlr && R"library(tlrmvnmvt)"

    # Pre-generate test cases to ensure consistency
    R"set.seed(42)"

    Σ = random_spd(n_dim)
    k = quantile(Normal(), (1 + 0.25^(1 / n_dim)) / 2)
    a = -sqrt.(diag(Σ)) * k
    b = sqrt.(diag(Σ)) * k

    results = DataFrame(
        rep=Int[],
        method=String[],
        value=Float64[],
        time=Float64[],
        error=Union{Float64,Missing}[],
        est_error=Float64[]
    )

    # 1) pnorm (Julia/QMC)
    for rep in 1:n_reps

        val, err, t = run_pnorm(Σ, a, b;
            max_pts=Int(qmc_pts / 12), seed=seed + rep)
        push!(results, (rep=rep, method="pnorm", value=val, time=t, error=missing, est_error=err))
    end

    # 2) mvnormcdf (Julia)
    for rep in 1:n_reps

        val, err, t = run_mvnormcdf(Σ, a, b; m=qmc_pts, seed=seed + rep)
        push!(results, (rep=rep, method="mvnormcdf", value=val, time=t, error=missing, est_error=err))
    end

    # 3) tlr (R/tlrmvnmvt, optional)
    if has_tlr
        for rep in 1:n_reps

            val, err, t = run_tlrmvnmvt(a, b, Σ, Int(ceil(qmc_pts / 20)))
            push!(results, (rep=rep, method="tlr", value=val, time=t, error=missing, est_error=err))
        end
    end

    # Use pnorm as reference when available (matches existing preference order)
    baseline_method = any(results.method .== "pnorm") ? "pnorm" :
                      (any(results.method .== "mvnormcdf") ? "mvnormcdf" : "none")

    if baseline_method != "none"
        refs = Dict(row.rep => row.value for row in eachrow(results[results.method.==baseline_method, :]))
        for i in 1:nrow(results)
            if haskey(refs, results.rep[i])
                results.error[i] = abs(results.value[i] - refs[results.rep[i]])
            end
        end
    end

    return results, baseline_method
end

function summarize_results(df::DataFrame)
    if isempty(df)
        return DataFrame(), DataFrame()
    end

    # Group by method and calculate statistics for 'value' and 'est_error'
    val_summary = combine(groupby(df, :method),
        :value => median => :median,
        :value => std => :sd,
        :value => minimum => :min,
        :value => maximum => :max,
        :value => (x -> quantile(x, 0.25)) => :Q1,
        :value => (x -> quantile(x, 0.75)) => :Q3,
        :est_error => mean => :se_est
    )

    # Group by method and calculate statistics for 'time'
    time_summary = combine(groupby(df, :method),
        :time => median => :median,
        :time => minimum => :min,
        :time => maximum => :max,
        :time => (x -> quantile(x, 0.25)) => :Q1,
        :time => (x -> quantile(x, 0.75)) => :Q3
    )

    return val_summary, time_summary
end




function run_all_simulations(ns, n_ptss, n_reps, sim_fun=run_simulation)
    val_df = DataFrame()
    t_df = DataFrame()

    @showprogress for n in ns
        for n_pts in n_ptss
            results, baseline = sim_fun(n_dim=n, qmc_pts=n_pts, n_reps=n_reps)
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

ms = [Int(x) for x in simcfg("simulation_comparison", "m_values", [2^11 * 12, 2^11 * 120])]
small_n_powers = [Int(x) for x in simcfg("simulation_comparison", "small_n_powers", [4, 5, 6, 7, 8, 9])]
big_n_powers = [Int(x) for x in simcfg("simulation_comparison", "big_n_powers", [10, 11, 12])]


@time sim1 = run_all_simulations(2 .^ small_n_powers, ms, sim_reps)
@time sim2 = run_all_simulations(2 .^ big_n_powers, ms, sim_reps_big, run_simulation_big)

sim1_vals, sim1_times = sim1
sim2_vals, sim2_times = sim2

comparisson_vals_1 = copy(sim1_vals)
comparisson_vals_1[!, :sim] = fill("sim1", nrow(comparisson_vals_1))
comparisson_vals_2 = copy(sim2_vals)
comparisson_vals_2[!, :sim] = fill("sim2", nrow(comparisson_vals_2))
comparisson_vals = vcat(comparisson_vals_1, comparisson_vals_2; cols=:union)

comparisson_times_1 = copy(sim1_times)
comparisson_times_1[!, :sim] = fill("sim1", nrow(comparisson_times_1))
comparisson_times_2 = copy(sim2_times)
comparisson_times_2[!, :sim] = fill("sim2", nrow(comparisson_times_2))
comparisson_times = vcat(comparisson_times_1, comparisson_times_2; cols=:union)

CSV.write(sim_resultpath("comparisson_vals.csv"), comparisson_vals)
CSV.write(sim_resultpath("comparisson_times.csv"), comparisson_times)
