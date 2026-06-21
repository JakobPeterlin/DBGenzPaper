
include(joinpath(@__DIR__, "setup.jl"))
using DBGenzPaper
using LinearAlgebra, OpenBLAS_jll, Statistics, DataFrames, CSV, Random, Distributions
use_accelerated_blas!()

import ProgressMeter: @showprogress
using SpecialFunctions: besselk, gamma




function rand_spd(n::Int; jitter::Float64=1.0, rng=Random.default_rng())
    A = randn(rng, n, n)
    Σ = A * A'
    Σ += jitter * n * I # keep it well conditioned
    return Matrix(Symmetric(Σ))
end


function fixed_dense(n::Int; jitter::Float64=1.0, rng=Random.default_rng())
    M = ones(n, n) + jitter * I
    return M
end


function fixed_spd(n::Int; jitter::Float64=1.0, rng=Random.default_rng())
    return fixed_dense(n; jitter=jitter, rng=rng)
end



function fixed_sparse(n::Int, n_sp::Int)
    M = diagm(ones(n) * n_sp / 2)

    for i in 1:n_sp
        M[diagind(M, i)] .= 1.0
    end

    M = M + M'

    return M
end






sd(v::Vector{T}) where T = sqrt(var(v))





function mattern_cov(n; ν=1.5, dist=1.0, σ²=1.0)
    Σ = Matrix{Float64}(undef, n, n)
    c = σ² * 2^(1 - ν) / gamma(ν)

    for j in 1:n
        Σ[j, j] = σ²

        for i in (j+1):n
            z = sqrt(2ν) * (i - j) / dist
            v = c * z^ν * besselk(ν, z)
            Σ[i, j] = v
            Σ[j, i] = v
        end
    end

    return Σ
end

mattern_cov0(n) = mattern_cov(n; ν=1.5, dist=n * 0.01) # 1 and 0.86 nnz
mattern_cov1(n) = mattern_cov(n; ν=1.5, dist=n * 0.001) # 0.64 and 0.12 nnz
mattern_cov2(n) = mattern_cov(n; ν=1.5, dist=n * 0.0001) # 0.08 and 0.01 nnz