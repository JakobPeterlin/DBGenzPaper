using Sobol, Random, LinearAlgebra, SparseArrays, LoopVectorization
using SpecialFunctions: erf, erfinv
using MvNormalCDF

include("cholesky.jl")


struct QMCOpts{T}
    m::Int
    block_size_i::Int
    block_size_j::Int
    n_blocks::Int
    n_reps::Int
    n_bits::Int
    max_pts::Int
    max_abs_err::T
end


struct QMCDataSparse{T}
    C::CholeskyGenz{T}
    sobol_gen::SobolSeq
    X_F::Matrix{Float64}
    X0::Matrix{UInt64}
    X::Matrix{UInt64}
    a::Vector{T}
    b::Vector{T}
    shifts::Matrix{UInt64}
    Ys::Vector{Matrix{T}}
    sub_mats::Vector{Vector{T}}
    c_vecs::Vector{Vector{T}}
    dc_vecs::Vector{Vector{T}}
    p_vecs::Vector{Vector{T}}
    qmc_reps::Vector{T}
    sum_p_threads::Vector{T}
    qmc_opts::QMCOpts{T}
    U_S::SparseMatrixCSC{T,Int}
end



function QMC_opts(T=Float64;
    m::Integer=10^4,
    block_size_i=2^6,
    block_size_j::Int=2^6,
    n_blocks=Threads.nthreads(),
    n_reps::Int=12,
    n_bits::Int=32,
    max_pts::Int=10^4,
    max_abs_err=10^(-6))

    return QMCOpts{T}(m, block_size_i, block_size_j, n_blocks, n_reps, n_bits, max_pts, max_abs_err)
end




function QMCDataSparse(C::Matrix{T0},
    a::Vector{T0},
    b::Vector{T0};
    opts::QMCOpts{T}=QMC_opts(T)) where {T,T0}

    m = opts.m
    n_blocks = opts.n_blocks
    block_size_j = opts.block_size_j
    block_size_i = opts.block_size_i
    n_reps = opts.n_reps
    n_bits = opts.n_bits
    n = size(C, 1)
    s0 = SobolSeq(n - 1)
    s0 = skip(s0, 2 * m - 1)
    X_F = zeros(T, n - 1, m)
    sobol_mat!(X_F, s0)
    X0 = zeros(UInt64, n - 1, m)
    copy_as_UInt64!(X0, X_F)
    X = similar(X0)
    Ys = [zeros(block_size_j, n - 1) for i in 1:n_blocks]
    shifts = zeros(UInt64, n - 1, n_reps)
    sub_mats = [zeros(T, block_size_j) for i in 1:n_blocks]
    c_vecs = [zeros(T, block_size_j) for i in 1:n_blocks]
    dc_vecs = deepcopy(c_vecs)
    p_vecs = deepcopy(c_vecs)
    qmc_reps = zeros(T, n_reps)
    sum_p_threads = zeros(T, n_blocks)
    chol = T0 == T ? cholesky_classic!(copy(C), copy(a), copy(b)) : cholesky_classic!(convert.(T, C), convert.(T, a), convert.(T, b))

    U_S = sparse(triu(chol.U, 1))

    return QMCDataSparse{T}(chol, s0, X_F, X0, X, chol.a, chol.b, shifts, Ys, sub_mats, c_vecs, dc_vecs, p_vecs, qmc_reps, sum_p_threads, opts, U_S)
end



## Helper methods




function sobol_mat!(X, s)
    for j in axes(X, 2)
        next!(s, view(X, :, j))
    end

    return X
end






function copy_as_UInt64!(X, X_f, n_bits=32)
    scale = Float64(UInt64(1) << n_bits)
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            x_ij = X_f[i, j]
            x_ij = x_ij >= 1.0 ? prevfloat(1.0) : x_ij
            X[i, j] = floor(UInt64, floor(x_ij * scale))
        end
    end

    return X
end







function sd(v::Vector{T}, mean::T) where T
    n = length(v)
    result = zero(T)

    @inbounds for i in eachindex(v)
        result += (v[i] - mean)^2 / n
    end

    return sqrt(result / (n - 1))
end









@inline function normal_cdf(x::T) where T
    c = T(0.5)
    c2 = T(sqrt(2.0))
    return c * (1 + erf(x / c2))
end




@inline function qmc_loop!(D::QMCDataSparse{T}, n_pts0::Int, c_1::T, dc_1::T) where T
    n = length(D.b)
    L = D.qmc_opts.n_bits
    mask = (UInt64(1) << L) - 1
    ep0 = eps(T)
    ep1 = one(T) - eps(T)
    c05 = T(0.5)
    c2 = T(sqrt(2.0))
    c20 = T(2.0)
    c10 = T(1.0)
    c_min = T(9.0)
    c_max = T(-9.0)
    x_shift = ldexp(one(T), -L)

    for k in 1:length(D.qmc_reps)
        sum_p_threads = D.sum_p_threads
        fill!(sum_p_threads, zero(T))

        Threads.@threads for j1 in 1:D.qmc_opts.block_size_j:size(D.X, 2)
            j2 = min(j1 + D.qmc_opts.block_size_j - 1, size(D.X, 2))
            r_j = j1:j2
            i_t = mod(Threads.threadid(), Threads.nthreads()) + 1
            Y_j = D.Ys[i_t]
            p_j = D.p_vecs[i_t]
            c_j = D.c_vecs[i_t]
            dc_j = D.dc_vecs[i_t]
            s_j = D.sub_mats[i_t]
            U_S = D.U_S
            jlen = j2 - j1 + 1
            joff = j1 - 1

            fill!(c_j, c_1)
            fill!(s_j, zero(T))
            fill!(dc_j, dc_1)
            fill!(p_j, dc_1)

            for i1 in 1:D.qmc_opts.block_size_i:(n-1)
                i2 = min(i1 + D.qmc_opts.block_size_i - 1, n - 1)

                for i in i1:i2
                    # dimension in MVN is i+1; previous y columns are 1..i
                    shift_i = D.shifts[i, k] & mask

                    @turbo for i_b in 1:jlen
                        i_r = joff + i_b
                        y_i = T((D.X[i, i_r] ⊻ shift_i) & mask) * x_shift
                        p = c_j[i_b] + y_i * dc_j[i_b]
                        Y_j[i_b, i] = c20 * min(max(p, ep0), ep1) - c10
                    end

                    @simd for i_b in 1:jlen
                        Y_j[i_b, i] = c2 * erfinv(Y_j[i_b, i])
                    end

                    # Sparse multiply equivalent to:
                    #   s_j = view(Y_j, :, 1:i) * U_S[1:i, i+1]
                    # but without calling mul! inside the threaded region.
                    fill!(s_j, zero(T))
                    @inbounds for p_u in nzrange(U_S, i + 1)
                        i_u = U_S.rowval[p_u] # 1 <= i_u <= i (strict upper-triangular)
                        v_u = U_S.nzval[p_u]
                        @turbo for i_b in 1:jlen
                            s_j[i_b] = muladd(Y_j[i_b, i_u], v_u, s_j[i_b])
                        end
                    end

                    a_i = D.a[i+1]
                    b_i = D.b[i+1]
                    u_ii2 = one(T) / (D.C.U[i+1, i+1] * c2)

                    @inbounds @simd for i_b in 1:jlen
                        s_val = s_j[i_b]
                        c_ij = (a_i - s_val) * u_ii2
                        d_ij = (b_i - s_val) * u_ii2

                        c_j[i_b] = c_ij
                        dc_j[i_b] = d_ij
                    end

                    @turbo for i_b in 1:jlen
                        c_j[i_b] = erf(max(min(c_j[i_b], c_min), c_max))
                        dc_j[i_b] = erf(max(min(dc_j[i_b], c_min), c_max))
                    end

                    @turbo for i_b in 1:jlen
                        c_j[i_b] = c05 * (c10 + c_j[i_b])
                        dc_j[i_b] = c05 * (c10 + dc_j[i_b]) - c_j[i_b]
                    end

                    @inbounds @simd for i_b in 1:jlen
                        p_j[i_b] *= dc_j[i_b]
                    end
                end
            end

            sum_p_threads[i_t] += sum(view(p_j, 1:jlen))
        end

        n_pts_local = size(D.X, 2)
        n_pts_total = n_pts0 + n_pts_local
        mean_rep = sum(sum_p_threads) / n_pts_total
        D.qmc_reps[k] = D.qmc_reps[k] * (n_pts0 / n_pts_total) + mean_rep
    end
end







## Integration

function qmc_pnorm!(D::QMCDataSparse{T}, rng) where T
    n = length(D.b)
    L = D.qmc_opts.n_bits
    mask = (UInt64(1) << L) - 1
    # z_dist = Normal{T}(T(0.0), T(1.0)) # Unused
    ep0 = eps(T)
    ep1 = one(T) - eps(T)

    copyto!(D.X, D.X0)
    rand!(rng, D.shifts)

    fill!(D.qmc_reps, zero(T))
    n_reps = length(D.qmc_reps)

    arg_1_a = D.a[1] / (D.C.U[1, 1] * T(sqrt(2.0)))
    arg_1_b = D.b[1] / (D.C.U[1, 1] * T(sqrt(2.0)))
    c_1 = T(0.5) * (1 + erf(arg_1_a))
    dc_1 = T(0.5) * (erf(arg_1_b) - erf(arg_1_a))

    n_pts = 0

    qmc_loop!(D, n_pts, c_1, dc_1)
    result = sum(D.qmc_reps) / n_reps
    err_acc = 3 * sd(D.qmc_reps, result)
    n_pts += size(D.X_F, 2)

    max_pts = D.qmc_opts.max_pts
    max_abs_err = D.qmc_opts.max_abs_err

    if ((n_pts >= max_pts) | (err_acc < max_abs_err))
        return result, err_acc, n_pts
    end

    while ((n_pts < max_pts) & (err_acc > max_abs_err))
        sobol_mat!(D.X_F, D.sobol_gen)
        copy_as_UInt64!(D.X, D.X_F) # Update X with new points

        qmc_loop!(D, n_pts, c_1, dc_1)

        result = sum(D.qmc_reps) / n_reps
        err_acc = 3 * sd(D.qmc_reps, result)
        n_pts += size(D.X_F, 2)
    end

    return result, err_acc, n_pts
end






