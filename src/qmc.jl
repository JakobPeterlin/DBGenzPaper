using Sobol, Random, LinearAlgebra, LoopVectorization, SparseArrays


using Polyester

include("cholesky.jl")
include("qmc_generators.jl"), quantile, cdf

using MvNormalCDF
using Distributions: Normal



struct QMCOpts{T}
    chol_block_size::Int
    chol_block_size2::Int
    m::Int
    block_size_i::Int
    block_size_i2::Int
    block_size_j::Int
    n_blocks::Int
    n_reps::Int
    n_bits::Int
    max_pts::Int
    max_abs_err::T
end








struct QMCData{T,G<:QMCGenrator{T}}
    C::CholeskyGenz{T}
    a::Vector{T}
    b::Vector{T}
    qmc_gen::G
    Ys::Vector{Matrix{T}}
    sub_mats::Vector{Matrix{T}}
    c_vecs::Vector{Vector{T}}
    dc_vecs::Vector{Vector{T}}
    p_vecs::Vector{Vector{T}}
    qmc_reps::Vector{T}
    sum_p_threads::Vector{T}
    qmc_opts::QMCOpts{T}
end









struct QMCDataSparse{T,G<:QMCGenrator{T}}
    C::CholeskyGenz{T}
    a::Vector{T}
    b::Vector{T}
    qmc_gen::G
    Ys::Vector{Matrix{T}}
    sub_mats::Vector{Vector{T}}
    c_vecs::Vector{Vector{T}}
    dc_vecs::Vector{Vector{T}}
    p_vecs::Vector{Vector{T}}
    qmc_reps::Vector{T}
    sum_p_threads::Vector{T}
    qmc_opts::QMCOpts{T}
    U_S::SparseMatrixCSC{T,Int}
    nz_ranges::Vector{UnitRange{Int}}
end




struct QMCData_Distributions{T,G<:QMCGenrator{T}}
    C::CholeskyGenz{T}
    a::Vector{T}
    b::Vector{T}
    qmc_gen::G
    Ys::Vector{Matrix{T}}
    sub_mats::Vector{Matrix{T}}
    c_vecs::Vector{Vector{T}}
    dc_vecs::Vector{Vector{T}}
    p_vecs::Vector{Vector{T}}
    qmc_reps::Vector{T}
    sum_p_threads::Vector{T}
    qmc_opts::QMCOpts{T}

    Z::Normal{T}
end















function QMC_opts(T=Float64;
    chol_block_size::Int=2^6,
    chol_block_size2::Int=2^9,
    m::Integer=10^4,
    block_size_i=2^6,
    block_size_j::Int=2^6,
    block_size_i2::Int=2^5,
    n_blocks=Threads.nthreads(),
    n_reps::Int=12,
    n_bits::Int=32,
    max_pts::Int=10^4,
    max_abs_err=10^(-6))

    return QMCOpts{T}(chol_block_size, chol_block_size2, m, block_size_i, block_size_i2, block_size_j, n_blocks, n_reps, n_bits, max_pts, max_abs_err)
end







function truncate_matrixU!(U::Matrix{T}) where T
    min_val = eps(T) / T(size(U, 1) * 9)
    @batch minbatch = 2^6 for j in axes(U, 2)
        @inbounds for i in axes(U, 1)
            if abs(U[i, j]) < min_val
                U[i, j] = zero(T)
            end
        end
    end
end






function QMCData(C::Matrix{T0},
    a::Vector{T0},
    b::Vector{T0},
    opts::QMCOpts{T}=QMC_opts(T),
    rng=Random.default_rng(),
    qmc_type=:Richtmyer) where {T,T0}

    m = opts.m
    n_blocks = opts.n_blocks
    block_size_j = opts.block_size_j
    block_size_i = opts.block_size_i
    block_size_i2 = opts.block_size_i2
    n_reps = opts.n_reps
    n_bits = opts.n_bits
    n = size(C, 1)


    if n - 1 < opts.block_size_i
        block_size_i = n - 1

        if (n - 1) < opts.block_size_i2
            block_size_i2 = n - 1
        end

        opts = QMCOpts{T}(opts.chol_block_size, opts.chol_block_size2, m, block_size_i, block_size_i2, block_size_j, opts.n_blocks, opts.n_reps, opts.n_bits, opts.max_pts, opts.max_abs_err)
    end


    qmc_gen = if qmc_type == :Sobol
        SobolQMC(T, n - 1, m, n_reps, rng; n_bits=n_bits, skip0=(2 * m - 1))
    else
        RichtmyerQMC(T, n - 1, m, n_reps, rng)
    end
    Ys = [zeros(block_size_j, n - 1) for i in 1:n_blocks]
    sub_mats = [zeros(T, block_size_j, block_size_i) for i in 1:n_blocks]
    c_vecs = [zeros(T, block_size_j) for i in 1:n_blocks]
    dc_vecs = deepcopy(c_vecs)
    p_vecs = deepcopy(c_vecs)
    qmc_reps = zeros(T, n_reps)
    sum_p_threads = zeros(T, n_blocks)
    chol = if T0 == T
        cholesky_genz!(copy(C), copy(a), copy(b), opts.chol_block_size, opts.chol_block_size2)
    else
        cholesky_genz!(convert.(T, C), convert.(T, a), convert.(T, b), opts.chol_block_size, opts.chol_block_size2)
    end

    if chol.rank != n
        throw(ArgumentErr or("QMCData requires a full-rank Cholesky factorization."))
    end

    truncate_matrixU!(chol.U)

    return QMCData{T,typeof(qmc_gen)}(chol, chol.a, chol.b, qmc_gen, Ys, sub_mats, c_vecs, dc_vecs, p_vecs, qmc_reps, sum_p_threads, opts)
end







function QMCData_Distributions(D::QMCData{T,G}) where {T,G}
    return QMCData_Distributions{T,G}(
        D.C, D.a, D.b, D.qmc_gen, D.Ys, D.sub_mats, D.c_vecs, D.dc_vecs, D.p_vecs,
        D.qmc_reps, D.sum_p_threads, D.qmc_opts, Normal{T}(zero(T), one(T))
    )
end









function QMCDataSparse(C::Matrix{T0},
    a::Vector{T0},
    b::Vector{T0},
    opts::QMCOpts{T}=QMC_opts(T),
    rng=Random.default_rng(),
    qmc_type=:Sobol) where {T,T0}

    m = opts.m
    block_size_j = opts.block_size_j
    block_size_i = opts.block_size_i
    n_reps = opts.n_reps
    n_bits = opts.n_bits
    n = size(C, 1)

    if n - 1 < opts.block_size_i
        opts = QMCOpts{T}(opts.chol_block_size, opts.chol_block_size2, m, n, opts.block_size_i2, opts.block_size_j, opts.n_blocks, opts.n_reps, opts.n_bits, opts.max_pts, opts.max_abs_err)
        block_size_i = n - 1
    end

    # Ensure we have enough per-thread buffers for the threaded loop.
    n_threads = Threads.nthreads()
    n_blocks = max(opts.n_blocks, n_threads)
    opts_use = (n_blocks == opts.n_blocks) ? opts : QMCOpts{T}(opts.chol_block_size, opts.chol_block_size2, m, block_size_i, opts.block_size_i2, block_size_j, n_blocks, n_reps, n_bits, opts.max_pts, opts.max_abs_err)

    qmc_generator = if qmc_type == :Sobol
        SobolQMC(T, n - 1, m, n_reps, rng; n_bits=n_bits, skip0=(2 * m - 1))
    else
        RichtmyerQMC(T, n - 1, m, n_reps, rng)
    end

    Ys = [zeros(T, block_size_j, n - 1) for _ in 1:n_blocks]
    sub_mats = [zeros(T, block_size_j) for _ in 1:n_blocks]
    c_vecs = [zeros(T, block_size_j) for _ in 1:n_blocks]
    dc_vecs = deepcopy(c_vecs)
    p_vecs = deepcopy(c_vecs)
    qmc_reps = zeros(T, n_reps)
    sum_p_threads = zeros(T, n_blocks)
    chol = if T0 == T
        cholesky_classic!(copy(C), copy(a), copy(b), opts.chol_block_size, opts.chol_block_size2)
    else
        cholesky_classic!(convert.(T, C), convert.(T, a), convert.(T, b), opts.chol_block_size, opts.chol_block_size2)
    end

    U = chol.U

    truncate_matrixU!(U)

    U_S = sparse(triu(U, 1))
    nz_ranges = Vector{UnitRange{Int}}(undef, n)

    for i in 1:n
        nz_ranges[i] = U_S.colptr[i]:(U_S.colptr[i+1]-1)
    end

    return QMCDataSparse{T,typeof(qmc_generator)}(chol, chol.a, chol.b, qmc_generator, Ys, sub_mats, c_vecs, dc_vecs, p_vecs, qmc_reps, sum_p_threads, opts_use, U_S, nz_ranges)
end












## Helper methods







function sd(v::Vector{T}, mean::T) where T
    n = length(v)
    result = zero(T)

    @inbounds for i in eachindex(v)
        result += (v[i] - mean)^2 / n
    end

    return sqrt(result / (n - 1))
end


function qmc_loop!(D::QMCData{T,G}, n_pts0::Int, c_1::T, dc_1::T) where {T,G}
    n = length(D.b)
    opts = D.qmc_opts
    gen = D.qmc_gen
    ep0 = eps(T)
    ep1 = one(T) - eps(T)
    c05 = T(0.5)
    c2 = T(sqrt(2.0))
    c20 = T(2.0)
    c10 = T(1.0)
    c_min = T(9.0)
    c_max = T(-9.0)

    for k in 1:length(D.qmc_reps)
        sum_p_threads = D.sum_p_threads
        fill!(sum_p_threads, zero(T))

        @batch for j1 in 1:opts.block_size_j:opts.m
            j2 = min(j1 + opts.block_size_j - 1, opts.m)
            r_j = j1:j2
            i_t = mod(Threads.threadid(), Threads.nthreads()) + 1
            Y_j = D.Ys[i_t]
            p_j = D.p_vecs[i_t]
            c_j = D.c_vecs[i_t]
            dc_j = D.dc_vecs[i_t]
            s_j = D.sub_mats[i_t]

            fill!(c_j, c_1)
            fill!(s_j, zero(T))
            fill!(dc_j, dc_1)
            fill!(p_j, dc_1)

            for ii1 in 1:opts.block_size_i:(n-1)
                ii2 = min(ii1 + opts.block_size_i - 1, n - 1)

                if ii1 > 1
                    if ii2 - ii1 == opts.block_size_i - 1
                        # precompute contributions from previous dimensions 1:(ii1-1)
                        # for the whole block of target dimensions (ii1+1):(ii2+1)
                        mul!(s_j,
                            view(Y_j, :, 1:(ii1-1)),
                            view(D.C.U, 1:(ii1-1), (ii1+1):(ii2+1)))
                    else
                        mul!(view(s_j, :, 1:(ii2-ii1+1)),
                            view(Y_j, :, 1:(ii1-1)),
                            view(D.C.U, 1:(ii1-1), (ii1+1):(ii2+1)))
                    end
                end

                for i1 in ii1:opts.block_size_i2:ii2
                    i2 = min(i1 + opts.block_size_i2 - 1, ii2)

                    if i1 > ii1
                        # Add only the incremental contribution from the *previous* inner-block
                        # (so we don't repeatedly re-add 1:(i1-1), and we keep future blocks correct).
                        prev1 = max(ii1, i1 - opts.block_size_i2)
                        prev2 = i1 - 1
                        # Update all remaining target dimensions in the current outer block.
                        # Target dims are (i1+1):(ii2+1) which correspond to columns (i1-ii1+1):(ii2-ii1+1) in s_j.
                        mul!(view(s_j, :, (i1-ii1+1):(ii2-ii1+1)),
                            view(Y_j, :, prev1:prev2),
                            view(D.C.U, prev1:prev2, (i1+1):(ii2+1)),
                            one(T), one(T))
                    end



                    for i in i1:i2
                        # dimension in MVN is i+1; previous y columns are 1..i
                        rand_points!(Y_j, gen, r_j, i, k, c_j, dc_j, ep0, ep1)
                        @simd for i_b in 1:length(r_j)
                            Y_j[i_b, i] = c2 * erfinv(Y_j[i_b, i])
                        end

                        # accumulate contribution from current block dimensions to the mean
                        s_blk = view(s_j, :, (i1-ii1+1):(i2-ii1+1))
                        # Equivalent to:
                        # mul!(s_blk[:, i-i1+1], Y_j[:, i1:i], D.C.U[i1:i, i+1], one(T), one(T))
                        # i.e. s_blk[:,col] .+= Y_j[:,i1:i] * u, but done explicitly for speed.
                        col = i - i1 + 1
                        u_i = view(D.C.U, i1:i, i + 1)
                        K = i - i1 + 1
                        @turbo for i_b in 1:length(r_j)
                            acc = s_blk[i_b, col]
                            for kk in 1:K
                                acc = muladd(Y_j[i_b, i1+kk-1], u_i[kk], acc)
                            end
                            s_blk[i_b, col] = acc
                        end

                        a_i = D.a[i+1]
                        b_i = D.b[i+1]
                        inv_u_ii2 = one(T) / (D.C.U[i+1, i+1] * c2) # Optimization: Precompute Inverse

                        @turbo for i_b in 1:length(r_j)
                            # Load
                            s_val = s_blk[i_b, i-i1+1]

                            # Calc (Mult instead of Div)
                            c_val = (a_i - s_val) * inv_u_ii2
                            dc_val = (b_i - s_val) * inv_u_ii2

                            # Erf
                            c_val = erf(max(min(c_val, c_min), c_max))
                            dc_val = erf(max(min(dc_val, c_min), c_max))

                            # Scale & Probability
                            c_val = c05 * (c10 + c_val)
                            dc_val = c05 * (c10 + dc_val) - c_val


                            c_j[i_b] = c_val
                            dc_j[i_b] = dc_val
                            p_j[i_b] *= dc_val
                        end
                    end
                end
            end

            sum_p_threads[i_t] += sum(view(p_j, 1:length(r_j)))
        end

        n_pts_local = opts.m
        n_pts_total = n_pts0 + n_pts_local
        mean_rep = sum(sum_p_threads) / n_pts_total
        D.qmc_reps[k] = D.qmc_reps[k] * (n_pts0 / n_pts_total) + mean_rep
    end
end








































function qmc_loop!(D::QMCDataSparse{T,G}, n_pts0::Int, c_1::T, dc_1::T) where {T,G}
    n = length(D.b)
    opts = D.qmc_opts
    gen = D.qmc_gen
    ep0 = eps(T)
    ep1 = one(T) - eps(T)
    c05 = T(0.5)
    c2 = T(sqrt(2.0))
    c20 = T(2.0)
    c10 = T(1.0)
    c_min = T(9.0)
    c_max = T(-9.0)

    for k in 1:length(D.qmc_reps)
        sum_p_threads = D.sum_p_threads
        fill!(sum_p_threads, zero(T))

        @batch for j1 in 1:opts.block_size_j:opts.m
            j2 = min(j1 + opts.block_size_j - 1, opts.m)
            r_j = j1:j2
            i_t = mod(Threads.threadid(), Threads.nthreads()) + 1
            Y_j = D.Ys[i_t]
            p_j = D.p_vecs[i_t]
            c_j = D.c_vecs[i_t]
            dc_j = D.dc_vecs[i_t]
            s_j = D.sub_mats[i_t]
            U_S = D.U_S
            jlen = j2 - j1 + 1

            fill!(c_j, c_1)
            fill!(s_j, zero(T))
            fill!(dc_j, dc_1)
            fill!(p_j, dc_1)

            for i1 in 1:opts.block_size_i:(n-1)
                i2 = min(i1 + opts.block_size_i - 1, n - 1)

                for i in i1:i2
                    # Sample current dimension i (used to update bounds for i+1)
                    rand_points!(Y_j, gen, r_j, i, k, c_j, dc_j, ep0, ep1)

                    @simd for i_b in 1:jlen
                        Y_j[i_b, i] = c2 * erfinv(Y_j[i_b, i])
                    end

                    # Sparse multiply equivalent to:
                    #   s_j = view(Y_j, :, 1:i) * U_S[1:i, i+1]
                    # but without calling mul! inside the threaded region.
                    fill!(s_j, zero(T))
                    nz_range_i = D.nz_ranges[i+1]

                    @turbo for p_u in nz_range_i
                        i_u = U_S.rowval[p_u] # 1 <= i_u <= i (strict upper-triangular)
                        v_u = U_S.nzval[p_u]
                        for i_b in 1:jlen
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

        n_pts_local = opts.m
        n_pts_total = n_pts0 + n_pts_local
        mean_rep = sum(sum_p_threads) / n_pts_total
        D.qmc_reps[k] = D.qmc_reps[k] * (n_pts0 / n_pts_total) + mean_rep
    end
end










## Integration



















































## Integration

function qmc_pnorm!(D::Union{QMCData{T},QMCDataSparse{T}}, use_AppleBLAS=!use_MKL_instead_of_ACC) where T
    n = length(D.b)
    gen = D.qmc_gen

    fill!(D.qmc_reps, zero(T))
    n_reps = length(D.qmc_reps)

    b_t = BLAS.get_num_threads()
    if !use_AppleBLAS && size(D.C.U, 1)^2 * D.qmc_opts.m > 2^12
        BLAS.set_num_threads(1)
    end



    arg_1_a = D.a[1] / (D.C.U[1, 1] * T(sqrt(2.0)))
    arg_1_b = D.b[1] / (D.C.U[1, 1] * T(sqrt(2.0)))
    c_1 = T(0.5) * (1 + erf(arg_1_a))
    dc_1 = T(0.5) * (erf(arg_1_b) - erf(arg_1_a))

    n_pts = 0

    qmc_loop!(D, n_pts, c_1, dc_1)
    result = sum(D.qmc_reps) / n_reps
    err_acc = 3 * sd(D.qmc_reps, result)
    n_pts += D.qmc_opts.m




    max_pts = D.qmc_opts.max_pts
    max_abs_err = D.qmc_opts.max_abs_err

    if ((n_pts >= max_pts) | (err_acc < max_abs_err))
        if !use_AppleBLAS && size(D.C.U, 1)^2 * D.qmc_opts.m > 2^12
            BLAS.set_num_threads(b_t)
        end

        return result, err_acc, n_pts
    end




    while ((n_pts < max_pts) & (err_acc > max_abs_err))
        next_points!(gen)

        qmc_loop!(D, n_pts, c_1, dc_1)

        result = sum(D.qmc_reps) / n_reps
        err_acc = 3 * sd(D.qmc_reps, result)
        n_pts += D.qmc_opts.m
    end

    if !use_AppleBLAS && size(D.C.U, 1)^2 * D.qmc_opts.m > 2^12
        BLAS.set_num_threads(b_t)
    end

    return result, err_acc, n_pts
end







