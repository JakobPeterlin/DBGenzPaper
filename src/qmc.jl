using Sobol, Random, LinearAlgebra, LoopVectorization, SparseArrays


include("cholesky.jl")
include("qmc_generators.jl")

using MvNormalCDF



struct QMCOpts{T}
    chol_block_size::Int
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








struct QMCData{T}
    C::CholeskyGenz{T}
    a::Vector{T}
    b::Vector{T}
    qmc_gen::QMCGenrator{T}
    Ys::Vector{Matrix{T}}
    sub_mats::Vector{Matrix{T}}
    c_vecs::Vector{Vector{T}}
    dc_vecs::Vector{Vector{T}}
    p_vecs::Vector{Vector{T}}
    qmc_reps::Vector{T}
    sum_p_threads::Vector{T}
    qmc_opts::QMCOpts{T}
end




struct QMCDataLowRank{T}
    C::CholeskyGenz{T}
    a::Vector{T}
    b::Vector{T}
    qmc_gen::QMCGenrator{T}
    Ys::Vector{Matrix{T}}
    sub_mats::Vector{Matrix{T}}
    c_vecs::Vector{Vector{T}}
    dc_vecs::Vector{Vector{T}}
    p_vecs::Vector{Vector{T}}
    qmc_reps::Vector{T}
    sum_p_threads::Vector{T}
    qmc_opts::QMCOpts{T}

    i_ranges::Vector{UnitRange{Int}}
    k_ranges::Vector{UnitRange{Int}}
    c_temp_vecs::Vector{Vector{T}}
    d_temp_vecs::Vector{Vector{T}}
    block_size_i::Int
end








struct QMCData_TB{T}
    C::CholeskyGenz{T}
    a::Vector{T}
    b::Vector{T}
    qmc_gen::QMCGenrator{T}
    Y::Matrix{T}
    Ys::Vector{Matrix{T}}
    sub_mat::Matrix{T}
    sub_mats::Vector{Matrix{T}}
    c_vec::Vector{T}
    dc_vec::Vector{T}
    p_vec::Vector{T}
    qmc_reps::Vector{T}
    qmc_opts::QMCOpts{T}
end




struct QMCData_TBLowRank{T}
    C::CholeskyGenz{T}
    a::Vector{T}
    b::Vector{T}
    qmc_gen::QMCGenrator{T}
    Y::Matrix{T}
    Ys::Vector{Matrix{T}}
    sub_mat::Matrix{T}
    sub_mats::Vector{Matrix{T}}
    c_vec::Vector{T}
    dc_vec::Vector{T}
    p_vec::Vector{T}
    qmc_reps::Vector{T}
    qmc_opts::QMCOpts{T}

    i_ranges::Vector{UnitRange{Int}}
    k_ranges::Vector{UnitRange{Int}}
    c_temp_vecs::Vector{Vector{T}}
    d_temp_vecs::Vector{Vector{T}}
    block_size_i::Int
end











struct QMCDataSparse{T}
    C::CholeskyGenz{T}
    a::Vector{T}
    b::Vector{T}
    qmc_gen::QMCGenrator{T}
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
    chol_block_size::Int=2^6,
    m::Integer=10^4,
    block_size_i=2^6,
    block_size_j::Int=2^6,
    block_size_i2::Int=2^5,
    n_blocks=Threads.nthreads(),
    n_reps::Int=12,
    n_bits::Int=32,
    max_pts::Int=10^4,
    max_abs_err=10^(-6))

    return QMCOpts{T}(chol_block_size, m, block_size_i, block_size_i2, block_size_j, n_blocks, n_reps, n_bits, max_pts, max_abs_err)
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

        opts = QMCOpts{T}(opts.chol_block_size, m, block_size_i, block_size_i2, block_size_j, opts.n_blocks, opts.n_reps, opts.n_bits, opts.max_pts, opts.max_abs_err)
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
        cholesky_genz!(copy(C), copy(a), copy(b), opts.chol_block_size)
    else
        cholesky_genz!(convert.(T, C), convert.(T, a), convert.(T, b), opts.chol_block_size)
    end

    if chol.rank == n
        return QMCData{T}(chol, chol.a, chol.b, qmc_gen, Ys, sub_mats, c_vecs, dc_vecs, p_vecs, qmc_reps, sum_p_threads, opts)
    else
        kj_s, _ = adapt_low_rank!(chol.U, a, b, chol.rank, chol.perm, eps(T))
        kj_s = [0 kj_s]
        i_ranges = [(kj_s[i]+1):kj_s[i+1] for i in 1:(length(kj_s)-1)]
        r_i_max = maximum(diff(kj_s))
        block_size_i = max(opts.block_size_i, r_i_max)
        k1 = 1
        k2 = 1
        k_ranges = Vector{UnitRange{Int}}(undef, 0)


        k = 2
        while k2 < (length(kj_s) - 1)
            if kj_s[k] - kj_s[k-1] > block_size_i
                k2 = k
                push!(k_ranges, k1:(k2-1))
                k1 = k2
            end
            k += 1
        end
        push!(k_ranges, k1:length(kj_s))


        sub_mats = [zeros(T, block_size_j, block_size_i) for i in 1:n_blocks]
        opts_lr = QMCOpts{T}(opts.m, block_size_i, opts.block_size_i2, opts.block_size_j, opts.n_blocks, opts.n_reps, opts.n_bits, opts.max_pts, opts.max_abs_err)

        return QMCDataLowRank{T}(chol, chol.a, chol.b, qmc_gen, Ys, sub_mats, c_vecs, dc_vecs, p_vecs, qmc_reps, sum_p_threads, opts_lr, i_ranges, k_ranges, deepcopy(c_vecs), deepcopy(c_vecs), block_size_i)
    end
end
















function QMCData_TB(C::Matrix{T0},
    a::Vector{T0},
    b::Vector{T0},
    opts::QMCOpts{T}=QMC_opts(T),
    rng=Random.default_rng(), qmc_type=:Sobol) where {T,T0}

    m = opts.m
    n_threads = Threads.nthreads()
    n_blocks = max(opts.n_blocks, n_threads)
    block_size_j = opts.block_size_j
    block_size_i = opts.block_size_i
    block_size_i2 = opts.block_size_i2
    n_reps = opts.n_reps
    n_bits = opts.n_bits
    n = size(C, 1)

    if n - 1 < opts.block_size_i
        opts = QMCOpts{T}(m, n, opts.block_size_i2, opts.block_size_j, opts.n_blocks, opts.n_reps, opts.n_bits, opts.max_pts, opts.max_abs_err)
        block_size_i = n - 1
    end
    qmc_gen = if qmc_type == :Sobol
        SobolQMC(T, n - 1, m, n_reps, rng; n_bits=n_bits, skip0=(2 * m - 1))
    else
        RichtmyerQMC(T, n - 1, m, n_reps, rng)
    end

    Y = zeros(T, m, n - 1)
    sub_mat = zeros(T, m, block_size_i)
    c_vec = zeros(T, m)
    dc_vec = deepcopy(c_vec)
    p_vec = deepcopy(c_vec)
    qmc_reps = zeros(T, n_reps)
    chol = if T0 == T
        cholesky_genz!(copy(C), copy(a), copy(b), opts.chol_block_size)
    else
        cholesky_genz!(convert.(T, C), convert.(T, a), convert.(T, b), opts.chol_block_size)
    end
    opts_use = (n_blocks == opts.n_blocks) ? opts : QMCOpts{T}(m, block_size_i, block_size_i2, block_size_j, n_blocks, n_reps, n_bits, opts.max_pts, opts.max_abs_err)

    if chol.rank == n
        Ys = [zeros(T, block_size_j, block_size_i2) for _ in 1:n_blocks]
        sub_mats = [zeros(T, block_size_j, block_size_i2) for _ in 1:n_blocks]
        return QMCData_TB{T}(chol, chol.a, chol.b, qmc_gen, Y, Ys, sub_mat, sub_mats, c_vec, dc_vec, p_vec, qmc_reps, opts_use)
    else
        # Low-rank path keeps the original (wide) per-thread buffers.
        Ys = [zeros(T, block_size_j, n - 1) for _ in 1:n_blocks]
        kj_s, _ = adapt_low_rank!(chol.U, a, b, chol.rank, chol.perm, eps(T))
        kj_s = [0 kj_s]
        i_ranges = [(kj_s[i]+1):kj_s[i+1] for i in 1:(length(kj_s)-1)]
        r_i_max = maximum(diff(kj_s))
        block_size_i = max(opts.block_size_i, r_i_max)
        k_ranges = Vector{UnitRange{Int}}(undef, 0)
        k1 = 1
        acc = 0
        @inbounds for k in 1:length(i_ranges)
            len_k = length(i_ranges[k])
            if (acc + len_k > block_size_i) && (k > k1)
                push!(k_ranges, k1:(k-1))
                k1 = k
                acc = 0
            end
            acc += len_k
        end
        push!(k_ranges, k1:length(i_ranges))


        sub_mat = zeros(T, m, block_size_i)
        sub_mats = [zeros(T, block_size_j, block_size_i) for i in 1:n_blocks]
        c_temp_vecs = [zeros(T, block_size_j) for i in 1:n_blocks]
        d_temp_vecs = deepcopy(c_temp_vecs)
        opts_lr = QMCOpts{T}(m, block_size_i, block_size_i2, block_size_j, n_blocks, n_reps, n_bits, opts.max_pts, opts.max_abs_err)

        return QMCData_TBLowRank{T}(chol, chol.a, chol.b, qmc_gen, Y, Ys, sub_mat, sub_mats, c_vec, dc_vec, p_vec, qmc_reps, opts_lr, i_ranges, k_ranges, c_temp_vecs, d_temp_vecs, block_size_i)
    end
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
        opts = QMCOpts{T}(opts.chol_block_size, m, n, opts.block_size_i2, opts.block_size_j, opts.n_blocks, opts.n_reps, opts.n_bits, opts.max_pts, opts.max_abs_err)
        block_size_i = n - 1
    end

    # Ensure we have enough per-thread buffers for the threaded loop.
    n_threads = Threads.nthreads()
    n_blocks = max(opts.n_blocks, n_threads)
    opts_use = (n_blocks == opts.n_blocks) ? opts : QMCOpts{T}(m, block_size_i, opts.block_size_i2, block_size_j, n_blocks, n_reps, n_bits, opts.max_pts, opts.max_abs_err)

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
        cholesky_classic!(copy(C), copy(a), copy(b), opts.chol_block_size)
    else
        cholesky_classic!(convert.(T, C), convert.(T, a), convert.(T, b), opts.chol_block_size)
    end

    U_S = sparse(triu(chol.U, 1))

    return QMCDataSparse{T}(chol, chol.a, chol.b, qmc_generator, Ys, sub_mats, c_vecs, dc_vecs, p_vecs, qmc_reps, sum_p_threads, opts_use, U_S)
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


function qmc_loop!(D::QMCData{T}, n_pts0::Int, c_1::T, dc_1::T) where T
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

        Threads.@threads for j1 in 1:opts.block_size_j:opts.m
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



















function qmc_loop!(D::QMCDataLowRank{T}, n_pts0::Int, c_1::T, dc_1::T) where T

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

        Threads.@threads for j1 in 1:opts.block_size_j:opts.m
            j2 = min(j1 + opts.block_size_j - 1, opts.m)
            r_j = j1:j2
            i_t = mod(Threads.threadid(), Threads.nthreads()) + 1
            Y_j = D.Ys[i_t]
            p_j = D.p_vecs[i_t]
            c_j = D.c_vecs[i_t]
            dc_j = D.dc_vecs[i_t]
            s_j = D.sub_mats[i_t]
            c_temp_vec = D.c_temp_vecs[i_t]
            d_temp_vec = D.d_temp_vecs[i_t]

            fill!(c_j, c_1)
            fill!(s_j, zero(T))
            fill!(dc_j, dc_1)
            fill!(p_j, dc_1)

            for k_r in D.k_ranges
                i1 = D.i_ranges[k_r[1]][1]
                i2 = D.i_ranges[k_r[end]][end]
                fill!(c_temp_vec, c_1)
                fill!(d_temp_vec, dc_1)

                if k_r[1] > 1
                    k1 = k_r[1] - 1

                    if i2 - i1 == D.block_size_i - 1
                        mul!(s_j,
                            view(Y_j, :, 1:k1),
                            view(D.C.U, 1:k1, (i1+1):(i2+1)))
                    else
                        mul!(view(s_j, :, 1:(i2-i1+1)),
                            view(Y_j, :, 1:k1),
                            view(D.C.U, 1:k1, (i1+1):(i2+1)))
                    end
                end


                for k_block in k_r

                    for i in D.i_ranges[k_block]
                        # dimension in MVN is i+1; previous y columns are 1..i
                        rand_points!(Y_j, gen, r_j, i, k)
                        @turbo for i_b in 1:length(r_j)
                            p = c_j[i_b] + Y_j[i_b, i] * dc_j[i_b]
                            p = min(max(p, ep0), ep1)
                            Y_j[i_b, i] = c2 * erfinv(c20 * p - c10)
                        end

                        # accumulate contribution from current block dimensions to the mean
                        mul!(view(s_j, :, i - i1 + 1),
                            view(Y_j, :, i1:i),
                            view(D.C.U, i1:i, i + 1),
                            one(T), one(T))

                        a_i = D.a[i+1]
                        b_i = D.b[i+1]
                        u_ii = D.C.U[i+1, i+1]

                        u_ii2 = (u_ii * c2)

                        @turbo for i_b in 1:length(r_j)
                            s_val = s_j[i_b, i-i1+1]
                            c_ij = (a_i - s_val) / u_ii2
                            d_ij = (b_i - s_val) / u_ii2

                            c_ij = erf(max(min(c_ij, c_min), c_max))
                            d_ij = erf(max(min(d_ij, c_min), c_max))

                            c_ij = c05 * (c10 + c_ij)
                            d_ij = c05 * (c10 + d_ij)

                            c_temp_vec[i_b] = max(c_temp_vec[i_b], c_ij)
                            d_temp_vec[i_b] = min(d_temp_vec[i_b], d_ij)
                        end

                    end

                    @turbo for i_b in 1:length(r_j)
                        c_j[i_b] = c_temp_vec[i_b]
                        dc_j[i_b] = d_temp_vec[i_b] - c_temp_vec[i_b]
                    end
                    @inbounds @simd for i_b in 1:length(r_j)
                        p_j[i_b] *= dc_j[i_b]
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





















function qmc_loop!(D::QMCDataSparse{T}, n_pts0::Int, c_1::T, dc_1::T) where T
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

        Threads.@threads for j1 in 1:opts.block_size_j:opts.m
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
                    @turbo for p_u in nzrange(U_S, i + 1)
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








function qmc_loop!(D::QMCData_TB{T}, n_pts0::Int, c_1::T, dc_1::T) where T
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
    s_mat = D.sub_mat
    Y_mat = D.Y
    p_vec = D.p_vec
    c_vec = D.c_vec
    dc_vec = D.dc_vec

    @inbounds for k in eachindex(D.qmc_reps)
        fill!(s_mat, zero(T))
        fill!(p_vec, dc_1)
        fill!(c_vec, c_1)
        fill!(dc_vec, dc_1)


        for i1 in 1:opts.block_size_i:(n-1)
            i2 = min(i1 + opts.block_size_i - 1, n - 1)
            iblock_len = i2 - i1 + 1

            if i1 > 1
                if iblock_len == opts.block_size_i
                    # precompute contributions from previous dimensions 1:(i1-1)
                    # for the whole block of target dimensions (i1+1):(i2+1)
                    mul!(s_mat,
                        view(Y_mat, :, 1:(i1-1)),
                        view(D.C.U, 1:(i1-1), (i1+1):(i2+1)))
                else
                    mul!(view(s_mat, :, 1:iblock_len),
                        view(Y_mat, :, 1:(i1-1)),
                        view(D.C.U, 1:(i1-1), (i1+1):(i2+1)))
                end
            end


            Threads.@threads for j1 in 1:opts.block_size_j:opts.m
                j2 = min(j1 + opts.block_size_j - 1, opts.m)
                buf_idx = Threads.threadid() % Threads.nthreads() + 1
                Y_block = D.Ys[buf_idx]
                s_block = D.sub_mats[buf_idx]
                n_j = j2 - j1 + 1
                j_off = j1 - 1

                for i_sub_start in i1:opts.block_size_i2:i2
                    i_sub_end = min(i_sub_start + opts.block_size_i2 - 1, i2)
                    n_i_sub = i_sub_end - i_sub_start + 1

                    @inbounds for col in 1:n_i_sub
                        src_col = (i_sub_start - i1) + col
                        @turbo for jj in 1:n_j
                            s_block[jj, col] = s_mat[j_off+jj, src_col]
                        end
                    end

                    if i_sub_start > i1
                        mul!(view(s_block, 1:n_j, 1:n_i_sub),
                            view(Y_mat, (j_off+1):(j_off+n_j), i1:(i_sub_start-1)),
                            view(D.C.U, i1:(i_sub_start-1), (i_sub_start+1):(i_sub_end+1)),
                            one(T), one(T))
                    end

                    for i in i_sub_start:i_sub_end
                        i_local = i - i_sub_start + 1
                        rand_points!(Y_block, gen, j1:j2, i, i_local, k, c_vec, dc_vec, ep0, ep1)
                        @simd for j_local in 1:n_j
                            Y_block[j_local, i_local] = c2 * erfinv(Y_block[j_local, i_local])
                        end

                        @turbo for j_local in 1:n_j
                            s_val = s_block[j_local, i_local]
                            for t_local in 1:i_local
                                t = i_sub_start + t_local - 1
                                s_val = muladd(Y_block[j_local, t_local], D.C.U[t, i+1], s_val)
                            end
                            s_block[j_local, i_local] = s_val
                        end

                        a_i = D.a[i+1]
                        b_i = D.b[i+1]
                        inv_u_ii2 = one(T) / (D.C.U[i+1, i+1] * c2)

                        @turbo for j_local in 1:n_j
                            s_val = s_block[j_local, i_local]
                            c_val = (a_i - s_val) * inv_u_ii2
                            dc_val = (b_i - s_val) * inv_u_ii2
                            c_val = erf(max(min(c_val, c_min), c_max))
                            dc_val = erf(max(min(dc_val, c_min), c_max))
                            c_val = c05 * (c10 + c_val)
                            dc_val = c05 * (c10 + dc_val) - c_val
                            j = j_off + j_local
                            c_vec[j] = c_val
                            dc_vec[j] = dc_val
                            p_vec[j] *= dc_val
                        end
                    end

                    @inbounds for col in 1:n_i_sub
                        i = i_sub_start + col - 1
                        @turbo for jj in 1:n_j
                            Y_mat[j_off+jj, i] = Y_block[jj, col]
                        end
                    end
                end
            end

        end

        n_pts_local = opts.m
        n_pts_total = n_pts0 + n_pts_local
        mean_rep = sum(p_vec) / n_pts_total
        D.qmc_reps[k] = D.qmc_reps[k] * (n_pts0 / n_pts_total) + mean_rep
    end
end




















function qmc_loop!(D::QMCData_TBLowRank{T}, n_pts0::Int, c_1::T, dc_1::T) where T
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
    s_mat = D.sub_mat
    Y_mat = D.Y
    p_vec = D.p_vec
    c_vec = D.c_vec
    dc_vec = D.dc_vec

    @inbounds for k in eachindex(D.qmc_reps)
        fill!(s_mat, zero(T))
        fill!(p_vec, dc_1)
        fill!(c_vec, c_1)
        fill!(dc_vec, dc_1)

        for k_r in D.k_ranges
            i1 = D.i_ranges[k_r[1]][1]
            i2 = D.i_ranges[k_r[end]][end]
            iblock_len = i2 - i1 + 1

            if i1 > 1
                if iblock_len == opts.block_size_i
                    mul!(s_mat,
                        view(Y_mat, :, 1:(i1-1)),
                        view(D.C.U, 1:(i1-1), (i1+1):(i2+1)))
                else
                    mul!(view(s_mat, :, 1:iblock_len),
                        view(Y_mat, :, 1:(i1-1)),
                        view(D.C.U, 1:(i1-1), (i1+1):(i2+1)))
                end
            end

            Threads.@threads for j1 in 1:opts.block_size_j:opts.m
                j2 = min(j1 + opts.block_size_j - 1, opts.m)
                r_j = j1:j2
                i_t = mod(Threads.threadid(), Threads.nthreads()) + 1
                Y_j = D.Ys[i_t]
                s_j = D.sub_mats[i_t]
                c_temp_vec = D.c_temp_vecs[i_t]
                d_temp_vec = D.d_temp_vecs[i_t]
                jlen = j2 - j1 + 1
                @views c_j = view(c_vec, r_j)
                @views dc_j = view(dc_vec, r_j)
                @views p_j = view(p_vec, r_j)

                @inbounds for col in 1:iblock_len
                    copyto!(view(s_j, 1:jlen, col), view(s_mat, r_j, col))
                end

                for k_block in k_r
                    @turbo for i_b in 1:jlen
                        c_temp_vec[i_b] = c_j[i_b]
                        d_temp_vec[i_b] = c_j[i_b] + dc_j[i_b]
                    end

                    for i in D.i_ranges[k_block]
                        rand_points!(Y_j, gen, r_j, i, k, c_vec, dc_vec, ep0, ep1)
                        @simd for i_b in 1:jlen
                            Y_j[i_b, i] = c2 * erfinv(Y_j[i_b, i])
                        end

                        i_s = i - i1 + 1

                        @turbo for i_b in 1:jlen
                            s_val = s_j[i_b, i_s]
                            for i_j in 1:i_s
                                i_Y = i1 + i_j - 1
                                s_val = muladd(Y_j[i_b, i_Y], D.C.U[i_Y, i+1], s_val)
                            end
                            s_j[i_b, i_s] = s_val
                        end

                        a_i = D.a[i+1]
                        b_i = D.b[i+1]
                        u_ii2 = one(T) / (D.C.U[i+1, i+1] * c2)

                        @inbounds @simd for i_b in 1:jlen
                            s_val = s_j[i_b, i_s]
                            c_l = (a_i - s_val) * u_ii2
                            c_u = (b_i - s_val) * u_ii2
                            c_l = c05 * (c10 + erf(max(min(c_l, c_min), c_max)))
                            c_u = c05 * (c10 + erf(max(min(c_u, c_min), c_max)))
                            c_temp_vec[i_b] = max(c_temp_vec[i_b], c_l)
                            d_temp_vec[i_b] = min(d_temp_vec[i_b], c_u)
                        end
                    end

                    @turbo for i_b in 1:jlen
                        c_j[i_b] = c_temp_vec[i_b]
                        dc_j[i_b] = d_temp_vec[i_b] - c_temp_vec[i_b]
                    end

                    @inbounds @simd for i_b in 1:jlen
                        p_j[i_b] *= dc_j[i_b]
                    end
                end

                @inbounds for i in i1:i2
                    copyto!(view(Y_mat, r_j, i), view(Y_j, 1:jlen, i))
                end
            end

        end

        n_pts_local = opts.m
        n_pts_total = n_pts0 + n_pts_local
        mean_rep = sum(p_vec) / n_pts_total
        D.qmc_reps[k] = D.qmc_reps[k] * (n_pts0 / n_pts_total) + mean_rep
    end
end























## Integration

function qmc_pnorm!(D::Union{QMCData{T},QMCDataLowRank{T},QMCData_TB{T},QMCData_TBLowRank{T},QMCDataSparse{T}}, use_AppleBLAS=true) where T
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







