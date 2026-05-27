
include(joinpath(@__DIR__, "setup.jl"))
using DBGenzPaper
using LinearAlgebra, OpenBLAS_jll, Statistics, DataFrames, CSV, Random, Distributions
use_accelerated_blas!()

import ProgressMeter: @showprogress




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



function fixed_sparse(n::Int, n_sp::Int)
    M = diagm(ones(n) * n_sp / 2)

    for i in 1:n_sp
        M[diagind(M, i)] .= 1.0
    end

    M = M + M'

    return M
end






sd(v::Vector{T}) where T = sqrt(var(v))


