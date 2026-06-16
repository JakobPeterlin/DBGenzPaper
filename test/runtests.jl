using Test
using Random
using LinearAlgebra
using Distributions
using MvNormalCDF
using DBGenzPaper

function rand_spd(n::Int; jitter::Float64=4.0, rng=Random.default_rng())
    A = randn(rng, n, n)
    Σ = A * A'
    Σ += jitter * n * I
    return Matrix(Symmetric(Σ))
end

@testset "cholesky_genz!" begin
    Random.seed!(101)
    n = 2^10
    A = rand(n, n)
    C = A * A' + 100.0 * I
    a = -ones(n)
    b = ones(n)

    F = cholesky_genz!(copy(C), copy(a), copy(b))
    U = UpperTriangular(F.U)
    p = F.perm

    C_test = copy(C)
    d_test = [sqrt(C_test[i, i]) for i in 1:n]
    for i in 1:n
        if d_test[i] > 0
            C_test[:, i] /= d_test[i]
            C_test[i, :] /= d_test[i]
        end
    end
    @test maximum(abs, C_test[p, p] - U' * U) < 1e-10
end

@testset "cholesky_classic!" begin
    Random.seed!(102)
    n = 100
    A = rand(n, n)
    C = A * A' + 100.0 * I
    a = -ones(n)
    b = ones(n)

    C_test = copy(C)
    d_test = [sqrt(C_test[i, i]) for i in 1:n]
    for i in 1:n
        C_test[:, i] /= d_test[i]
        C_test[i, :] /= d_test[i]
    end

    for block_size in (8, 16, 32, 64)
        F = cholesky_classic!(copy(C), copy(a), copy(b), block_size, block_size, block_size, block_size)
        U = UpperTriangular(F.U)

        @test maximum(abs, C_test - U' * U) < 1e-10
    end
end

@testset "adapt_low_rank!" begin
    n = 30
    rank = 20
    M = zeros(n, n)

    for i in 1:rank
        M[i, i] = 1.0
    end

    for j in rank+1:n
        for i in 1:rank
            M[i, j] = rand()
        end
    end

    M[:, 25] .= 0.0

    a = zeros(n)
    b = ones(n)
    perm = collect(1:n)
    eps1 = 1e-10

    k_js, k_0 = adapt_low_rank!(M, a, b, rank, perm, eps1)

    @test k_0 == 1
    @test perm[end] == 25
    @test all(M[:, end] .== 0.0)
    @test length(k_js) == rank
end

@testset "qmc_pnorm! dense vs MvNormalCDF" begin
    rng = MersenneTwister(42)
    n_dim = 8
    Σ = rand_spd(n_dim; rng=rng)
    a = fill(-Inf, n_dim)
    b = fill(2.0, n_dim)

    opts = QMC_opts(
        Float64;
        m=2^11,
        max_pts=2^11 * 4,
        max_abs_err=0.0,
        block_size_i=2^8,
        block_size_i2=2^6,
        block_size_j=2^7,
    )

    val, err, _ = qmc_pnorm!(QMCData(copy(Σ), copy(a), copy(b), opts, MersenneTwister(7), :Richtmyer))
    val_ref, _ = mvnormcdf(Σ, a, b; m=2^14, rng=MersenneTwister(7))

    @test isfinite(val)
    @test isfinite(err)
    @test abs(val - val_ref) < 0.06
end

@testset "qmc_pnorm! sparse vs MvNormalCDF" begin
    n_dim = 8
    Σ = Matrix(1.8I, n_dim, n_dim)
    Σ[diagind(Σ, 1)] .= 0.25
    Σ[diagind(Σ, -1)] .= 0.25
    Σ = Symmetric(Σ) |> Matrix

    a = fill(-Inf, n_dim)
    b = fill(1.5, n_dim)

    opts = QMC_opts(
        Float64;
        m=2^11,
        max_pts=2^11 * 4,
        max_abs_err=0.0,
        block_size_i=2^6,
        block_size_i2=2^5,
        block_size_j=2^6,
    )

    val_sparse, err_sparse, _ = qmc_pnorm!(QMCDataSparse(copy(Σ), copy(a), copy(b), opts, MersenneTwister(21), :Richtmyer))
    val_ref, _ = mvnormcdf(Σ, a, b; m=2^14, rng=MersenneTwister(21))

    @test isfinite(val_sparse)
    @test isfinite(err_sparse)
    @test abs(val_sparse - val_ref) < 0.06
end
