using Random
using Distributions: Normal, cdf, quantile
using LoopVectorization: @turbo
using SpecialFunctions: erf, erfc, erfinv
using DataFrames: DataFrame
using CSV
using StatsBase

include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

function test_normal_cdf_accuracy(n::Integer; T=Float64, rng=Random.default_rng())
    x = randn(rng, T, n) * 2.0
    dist_cdf = similar(x)
    erf_cdf = similar(x)
    normal = Normal(zero(T), one(T))

    t_dist = @elapsed for i in eachindex(x)
        dist_cdf[i] = cdf(normal, x[i])
    end

    for i in eachindex(x)
        dist_cdf[i] = cdf(normal, BigFloat(x[i]))
    end


    c05 = T(0.5)
    c2 = (sqrt(T(0.5)))
    c10 = T(1.0)
    c_min = T(9.0)
    c_max = T(-9.0)

    t_erf = @elapsed @turbo for i in 1:length(x)
        x_i = x[i] * c2
        x_i = erf(max(min(x_i, c_min), c_max))
        erf_cdf[i] = c05 * (c10 + x_i)
    end


    erf_simd = similar(x)
    ci2 = -c2

    t_erf_simd = @elapsed begin

        @turbo for i in 1:length(x)
            erf_simd[i] = x[i] * ci2
        end

        @inbounds @simd for i in 1:length(erf_simd)
            erf_simd[i] = c05 * erfc(erf_simd[i])
        end
    end

    return (points=x, distributions=dist_cdf, qmc=erf_cdf, simd=erf_simd, t_dist=t_dist, t_erf=t_erf, t_erf_simd=t_erf_simd)
end

function test_normal_quantile_accuracy(n::Integer; T=Float64, rng=Random.default_rng())
    p = rand(rng, T, n)
    dist_quantile = similar(p)
    erfinv_quantile = similar(p)
    normal = Normal(zero(T), one(T))
    ep0 = eps(T)
    ep1 = one(T) - ep0

    t_dist = @elapsed for i in eachindex(p)
        p_i = min(max(p[i], ep0), ep1)
        dist_quantile[i] = quantile(normal, p_i)
    end

    c2 = T(sqrt(2.0))
    c20 = T(2.0)
    c10 = T(1.0)

    t_erf = @elapsed @simd for i in 1:length(p)
        p_i = min(max(p[i], ep0), ep1)
        erfinv_quantile[i] = c2 * erfinv(c20 * p_i - c10)
    end

    return (points=p, distributions=dist_quantile, qmc=erfinv_quantile, t_dist=t_dist, t_erf=t_erf)
end





function repeat_acc_tests(n::Integer, n_reps::Integer, T::Type=Float64; rng=Random.default_rng())
    max_max_error, max_error_sum, rmse_sum, ratio_sum = ntuple(_ -> zeros(T, 3), 4)

    for _ in 1:n_reps
        cdf_res = test_normal_cdf_accuracy(n; T=T, rng=rng)
        inv_res = test_normal_quantile_accuracy(n; T=T, rng=rng)

        errors = (
            abs.(cdf_res.qmc .- cdf_res.distributions),
            abs.(cdf_res.simd .- cdf_res.distributions),
            abs.(inv_res.qmc .- inv_res.distributions),
        )
        rep_max_error = maximum.(errors)

        max_max_error .= max.(max_max_error, rep_max_error)
        max_error_sum .+= rep_max_error
        rmse_sum .+= sqrt.(sum.(abs2, errors) ./ length.(errors))

        # Relative time quotient golden/candidate (>1 means candidate is faster).
        ratio_sum .+= (cdf_res.t_dist / cdf_res.t_erf, cdf_res.t_dist / cdf_res.t_erf_simd, inv_res.t_dist / inv_res.t_erf)
    end

    return DataFrame(
        func=["CDF", "CDF", "Quantile"],
        impl=["erf", "erfc", "erfinv"],
        max_max_error=max_max_error,
        mean_max_error=max_error_sum ./ n_reps,
        mean_rmse=rmse_sum ./ n_reps,
        ratio_mean=ratio_sum ./ n_reps,
    )
end




##


function run_erf_accuracy_tests()
    results = DataFrame[]

    for T in (Float64, Float32)
        df = repeat_acc_tests(2^5, 10^5, T)

        for col in (:max_max_error, :mean_max_error, :mean_rmse)
            df[!, col] ./= eps(T)
        end

        df[!, :Precision] = fill(string(T), size(df, 1))
        push!(results, df)
    end

    erf_acc = vcat(results...)
    CSV.write(sim_resultpath("erf_acc.csv"), erf_acc)

    return erf_acc
end

erf_acc = run_erf_accuracy_tests()
