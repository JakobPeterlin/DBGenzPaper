using CSV, DataFrames, Distributions, LinearAlgebra, Random, Statistics, ProgressMeter, Statistics, StatsBase, Dates, AppleAccelerate
include(joinpath(@__DIR__, "setup.jl"))
using DBGenzPaper

# We extend `Statistics.cov` with a method for `Asset`
import Statistics: cov

snp_data = CSV.read(datapath("data_snp.csv"), DataFrame)
aw_data = CSV.read(datapath("data_aw.csv"), DataFrame)

snp_n_miss = [count(!ismissing, snp_data[:, j]) for j in 1:size(snp_data, 2)]
aw_n_miss = [count(!ismissing, aw_data[:, j]) for j in 1:size(aw_data, 2)]

snp_data = snp_data[:, findall(snp_n_miss .> 1000)]
aw_data = aw_data[:, findall(aw_n_miss .> 1000)]

snp_data[:, 1] = Date.(snp_data[:, 1])
aw_data[:, 1] = Date.(aw_data[:, 1])

snp_data = sort!(snp_data, :date)
aw_data = sort!(aw_data, :date)

snp_n_uniques = [length(unique(snp_data[:, j])) for j in 1:size(snp_data, 2)]
aw_n_uniques = [length(unique(aw_data[:, j])) for j in 1:size(aw_data, 2)]

snp_data = snp_data[:, findall(snp_n_uniques .> 100)]
aw_data = aw_data[:, findall(aw_n_uniques .> 100)]



struct Asset{T}
    name::String
    indices::Vector{Int}
    log_returns::Vector{T}
    normalized::Vector{T}
end




function get_log_returns(prices, minl=-1.5, maxl=1.5)
    ii = findall(x -> (!ismissing(x) & !isnan(x) & !(x == 0.0)), prices)
    v = zeros(length(ii))

    for k in 2:length(ii)
        v[k] = log(prices[ii[k]] / prices[ii[k-1]])
    end

    iii = findall(x -> (x > minl) & (x < maxl), v)
    return ii[iii], v[iii]
end


function get_normalized_log_returns(lreturns)
    vals = invperm(sortperm(lreturns)) / (length(lreturns) + 1)
    return quantile.(Normal(), vals)
end





function Asset(name, prices)
    ixi, lrets = get_log_returns(prices)
    normalized = get_normalized_log_returns(lrets)
    return Asset(name, ixi, lrets, normalized)
end








function cov(a::Asset, b::Asset)
    val = 0.0
    n = 0
    i_b = 0
    k = 0
    n_b = length(b.indices)

    for (k_a, i_a) in enumerate(a.indices)
        while (i_b < i_a) & (k < n_b)
            k += 1
            i_b = b.indices[k]
        end

        if i_b == i_a
            val += a.normalized[k_a] * b.normalized[k]
            n += 1
        end
    end



    return val
end









get_assets(df) = [Asset(names(df)[j], df[:, j]) for j in 2:size(df, 2)]



function get_covariance_matrix(assets::Vector{Asset{T}}) where T
    n = length(assets)
    Σ = zeros(n, n)
    n_t = maximum(length(a.indices) for a in assets)

    Threads.@threads for j in 1:n
        for i in 1:j
            c_ij = cov(assets[i], assets[j]) / n_t
            Σ[i, j] = c_ij
            Σ[j, i] = c_ij
        end
    end
    return Σ
end






##

snp_assets = get_assets(snp_data)
aw_assets = get_assets(aw_data)


##

Σ_snp = get_covariance_matrix(snp_assets)
Σ_aw = get_covariance_matrix(aw_assets)


##

snp_cov = DataFrame(Σ_snp, :auto)
rename!(snp_cov, names(snp_data)[2:end])

aw_cov = DataFrame(Σ_aw, :auto)
rename!(aw_cov, names(aw_data)[2:end])



##

CSV.write(datapath("snp_cov.csv"), snp_cov)
CSV.write(datapath("aw_cov.csv"), aw_cov)


##











function get_bound(asset::Asset{T}, q, lower=true) where T
    i = 0

    if lower
        m = -Inf
        for j in eachindex(asset.indices)
            if (asset.log_returns[j] < q) & (asset.log_returns[j] > m)
                m = asset.log_returns[j]
                i = j
            end
        end

        if i != 0
            return asset.normalized[i]
        else
            return -Inf
        end
    else
        m = Inf

        for j in eachindex(asset.indices)
            if (asset.log_returns[j] > q) & (asset.log_returns[j] < m)
                m = asset.log_returns[j]
                i = j
            end
        end

        if i != 0
            return asset.normalized[i]
        else
            return Inf
        end
    end
end





function get_bounds(assets::Vector{Asset{T}}, alpha, lower=false) where T
    q = log(alpha)
    bounds_vec = zeros(T, length(assets))

    Threads.@threads for j in 1:length(assets)
        bounds_vec[j] = get_bound(assets[j], q, lower)
    end

    return bounds_vec
end



##


function integrate_bounds(assets::Vector{Asset{T}}, Σ::Matrix{T}, alpha, lower=true) where T
    bounds = get_bounds(assets, alpha, lower)
    opts = QMC_opts(m=2^11, max_pts=2^11 * 100, max_abs_err=0.0, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^7)
    data = QMCData(copy(Σ), -Inf * ones(size(Σ, 1)), bounds, opts, Random.default_rng())
    return qmc_pnorm!(data)
end


##
alphas = 1.0:0.025:1.3

ps_snp = [integrate_bounds(snp_assets, Σ_snp, alpha) for alpha in alphas]
ps_aw = [integrate_bounds(aw_assets, Σ_aw, alpha) for alpha in alphas]

##
# Build dataframe
ps_df = DataFrame(b=alphas)
ps_df.val_snp .= 0.0
ps_df.err_snp .= 0.0
ps_df.val_aw .= 0.0
ps_df.err_aw .= 0.0

for (i, b) in enumerate(alphas)
    ps_df.val_snp[i] = ps_snp[i][1]
    ps_df.err_snp[i] = ps_snp[i][2]
    ps_df.val_aw[i] = ps_aw[i][1]
    ps_df.err_aw[i] = ps_aw[i][2]
end

CSV.write(datapath("ps_aw.csv"), ps_df)
