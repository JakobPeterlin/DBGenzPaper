using Random
using Distributions: Normal, cdf, quantile
using LoopVectorization: @turbo
using SpecialFunctions: erf, erfinv


function test_normal_cdf_accuracy(n::Integer; T=Float64, rng=Random.default_rng())
    x = randn(rng, T, n)
    dist_cdf = similar(x)
    erf_cdf = similar(x)
    normal = Normal(zero(T), one(T))

    for i in eachindex(x)
        dist_cdf[i] = cdf(normal, x[i])
    end

    c05 = T(0.5)
    c2 = T(sqrt(2.0))
    c10 = T(1.0)
    c_min = T(9.0)
    c_max = T(-9.0)

    @turbo for i in 1:length(x)
        x_i = x[i] / c2
        x_i = erf(max(min(x_i, c_min), c_max))
        erf_cdf[i] = c05 * (c10 + x_i)
    end

    return (points=x, distributions=dist_cdf, qmc=erf_cdf)
end


function test_normal_quantile_accuracy(n::Integer; T=Float64, rng=Random.default_rng())
    p = rand(rng, T, n)
    dist_quantile = similar(p)
    erfinv_quantile = similar(p)
    normal = Normal(zero(T), one(T))

    ep0 = eps(T)
    ep1 = one(T) - eps(T)

    for i in eachindex(p)
        p_i = min(max(p[i], ep0), ep1)
        dist_quantile[i] = quantile(normal, p_i)
    end

    c2 = T(sqrt(2.0))
    c20 = T(2.0)
    c10 = T(1.0)

    @simd for i in 1:length(p)
        p_i = min(max(p[i], ep0), ep1)
        erfinv_quantile[i] = c2 * erfinv(c20 * p_i - c10)
    end

    return (points=p, distributions=dist_quantile, qmc=erfinv_quantile)
end




_, v1, v2 = test_normal_cdf_accuracy(10^4)
_, v3, v4 = test_normal_quantile_accuracy(10^4)


println(maximum(abs.(v1 - v2)))
println(maximum(abs.(v3 - v4)))