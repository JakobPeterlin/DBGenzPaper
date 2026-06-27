include(joinpath(@__DIR__, "setup.jl"))

using Sobol, Random, LinearAlgebra, LoopVectorization, TimerOutputs, Polyester, CSV, DataFrames
use_accelerated_blas!()

const to = TimerOutput()
const tos = [TimerOutput() for _ in 1:Threads.nthreads()]

include(joinpath(@__DIR__, "..", "..", "src", "cholesky.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "qmc_generators.jl"))

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

struct QMCData_timeit{T,G<:QMCGenrator{T}}
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

function QMCData_timeit(C::Matrix{T0},
    a::Vector{T0},
    b::Vector{T0};
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

    qmc_gen = @timeit to "Generating QMC points" begin
        if qmc_type == :Sobol
            SobolQMC(T, n - 1, m, n_reps, rng; n_bits=n_bits, skip0=(2 * m - 1))
        else
            RichtmyerQMC(T, n - 1, m, n_reps, rng)
        end
    end

    Ys = [zeros(block_size_j, n - 1) for _ in 1:n_blocks]
    sub_mats = [zeros(T, block_size_j, block_size_i) for _ in 1:n_blocks]
    c_vecs = [zeros(T, block_size_j) for _ in 1:n_blocks]
    dc_vecs = deepcopy(c_vecs)
    p_vecs = deepcopy(c_vecs)
    qmc_reps = zeros(T, n_reps)
    sum_p_threads = zeros(T, n_blocks)
    chol = @timeit to "Cholesky" begin
        if T0 == T
            cholesky_genz!(copy(C), copy(a), copy(b), opts.chol_block_size, opts.chol_block_size2)
        else
            cholesky_genz!(convert.(T, C), convert.(T, a), convert.(T, b), opts.chol_block_size, opts.chol_block_size2)
        end
    end

    if chol.rank != n
        throw(ArgumentError("QMCData_timeit requires a full-rank Cholesky factorization."))
    end

    truncate_matrixU!(chol.U)

    return QMCData_timeit{T,typeof(qmc_gen)}(chol, chol.a, chol.b, qmc_gen, Ys, sub_mats, c_vecs, dc_vecs, p_vecs, qmc_reps, sum_p_threads, opts)
end

function sd(v::Vector{T}, mean::T) where T
    n = length(v)
    result = zero(T)

    @inbounds for i in eachindex(v)
        result += (v[i] - mean)^2 / n
    end

    return sqrt(result / (n - 1))
end

@inline function qmc_loop!(D::QMCData_timeit{T,G}, n_pts0::Int, c_1::T, dc_1::T) where {T,G}
    n = length(D.b)
    opts = D.qmc_opts
    gen = D.qmc_gen
    ep0 = eps(T)
    ep1 = one(T) - eps(T)
    c05 = T(0.5)
    c2 = T(sqrt(2.0))
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

                @timeit tos[i_t] "BLAS mul!" if ii1 > 1
                    if ii2 - ii1 == opts.block_size_i - 1
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

                    @timeit tos[i_t] "BLAS mul!" if i1 > ii1
                        prev1 = max(ii1, i1 - opts.block_size_i2)
                        prev2 = i1 - 1
                        mul!(view(s_j, :, (i1-ii1+1):(ii2-ii1+1)),
                            view(Y_j, :, prev1:prev2),
                            view(D.C.U, prev1:prev2, (i1+1):(ii2+1)),
                            one(T), one(T))
                    end

                    for i in i1:i2
                        @timeit tos[i_t] "Affine scrambling" rand_points!(Y_j, gen, r_j, i, k, c_j, dc_j, ep0, ep1)
                        @timeit tos[i_t] "Computation of quantiles" @simd for i_b in 1:length(r_j)
                            Y_j[i_b, i] = c2 * erfinv(Y_j[i_b, i])
                        end

                        s_blk = view(s_j, :, (i1-ii1+1):(i2-ii1+1))
                        col = i - i1 + 1
                        u_i = view(D.C.U, i1:i, i + 1)
                        K = i - i1 + 1
                        @timeit tos[i_t] "Internal multiplication" @turbo for i_b in 1:length(r_j)
                            acc = s_blk[i_b, col]
                            for kk in 1:K
                                acc = muladd(Y_j[i_b, i1+kk-1], u_i[kk], acc)
                            end
                            s_blk[i_b, col] = acc
                        end

                        a_i = D.a[i+1]
                        b_i = D.b[i+1]
                        inv_u_ii2 = one(T) / (D.C.U[i+1, i+1] * c2)

                        @timeit tos[i_t] "Clculation of CDFs" @turbo for i_b in 1:length(r_j)
                            s_val = s_blk[i_b, i-i1+1]

                            c_val = (a_i - s_val) * inv_u_ii2
                            dc_val = (b_i - s_val) * inv_u_ii2

                            c_val = erf(max(min(c_val, c_min), c_max))
                            dc_val = erf(max(min(dc_val, c_min), c_max))

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

function qmc_pnorm!(D::QMCData_timeit{T}, rng, use_AppleBLAS=!use_MKL_instead_of_ACC) where T
    gen = D.qmc_gen

    @timeit to "Generating QMC points" reset_gen!(gen, rng)

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

    @timeit to "DB loop" qmc_loop!(D, n_pts, c_1, dc_1)
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
        @timeit to "Generating QMC points" next_points!(gen)

        @timeit to "DB loop" qmc_loop!(D, n_pts, c_1, dc_1)

        result = sum(D.qmc_reps) / n_reps
        err_acc = 3 * sd(D.qmc_reps, result)
        n_pts += D.qmc_opts.m
    end

    if !use_AppleBLAS && size(D.C.U, 1)^2 * D.qmc_opts.m > 2^12
        BLAS.set_num_threads(b_t)
    end

    return result, err_acc, n_pts
end

function timed_qmc_opts(max_pts::Int)
    return use_MKL_instead_of_ACC ? QMC_opts(Float64;
        chol_block_size=2^9, chol_block_size2=2^9, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^6) :
           QMC_opts(Float64; chol_block_size=2^5, chol_block_size2=2^7, m=max_pts, max_pts=max_pts, max_abs_err=0.0, block_size_i=2^10, block_size_i2=2^6, block_size_j=max_pts >= 2^12 ? 2^7 : 2^7)
end

function to_df(to::TimerOutput, total_time=0; subtract_cholesky_from_prealloc::Bool=true)
    flat = TimerOutputs.flatten(to)

    df = DataFrame(
        Section=collect(keys(flat.inner_timers)),
        Calls=[TimerOutputs.ncalls(t) for t in values(flat.inner_timers)],
        Time_sec=[TimerOutputs.time(t) / 1e9 for t in values(flat.inner_timers)],
        Alloc_MiB=[TimerOutputs.allocated(t) / 1024^2 for t in values(flat.inner_timers)])

    if subtract_cholesky_from_prealloc
        i_pre = findfirst(==("Pre-allocation"), df.Section)
        i_chol = findfirst(==("Cholesky"), df.Section)
        if !(isnothing(i_pre) || isnothing(i_chol))
            df.Time_sec[i_pre] = max(df.Time_sec[i_pre] - df.Time_sec[i_chol], 0.0)
            df.Alloc_MiB[i_pre] = max(df.Alloc_MiB[i_pre] - df.Alloc_MiB[i_chol], 0.0)
        end
    end

    total_time = total_time == 0 ? sum(df.Time_sec) : total_time

    df.Percent = df.Time_sec / total_time
    return df
end

function merge_timers(tos::AbstractVector{<:TimerOutput})
    merged = TimerOutput()
    for t in tos
        merge!(merged, t)
    end
    return merged
end

function combined_timer_df(
    to::TimerOutput,
    tos::AbstractVector{<:TimerOutput};
    drop_sections::Vector{String}=String["Total", "DB loop"],
)
    df_to = to_df(to, 0; subtract_cholesky_from_prealloc=true)
    if !isempty(drop_sections)
        keep = .!in.(df_to.Section, Ref(drop_sections))
        df_to = df_to[keep, :]
    end

    df_tos = to_df(merge_timers(tos), 0; subtract_cholesky_from_prealloc=false)
    n_tos = length(tos)
    if n_tos > 0
        df_tos.Time_sec ./= n_tos
        df_tos.Alloc_MiB ./= n_tos
    end

    df = vcat(df_to, df_tos)
    total = sum(df.Time_sec)
    df.Percent = total == 0 ? zero.(df.Time_sec) : (df.Time_sec ./ total)
    return df
end

n_ps = [2^4, 2^6, 2^8, 2^10, 2^12]
n_reps = simcfg("simulation_timed", "n_reps", 10)
b0 = Float64(simcfg("simulation_timed", "b0", 3.0))
m_values = [2^11, 2^11 * 10]

function reset_all_timers!()
    reset_timer!(to)
    for to_i in tos
        reset_timer!(to_i)
    end
end

function trial_run!(M, n_p, b0, opts, rng=Random.default_rng())
    data = QMCData_timeit(copy(M), -Inf * ones(n_p), b0 * ones(n_p); opts=opts)
    qmc_pnorm!(data, rng)
end

# Small untimed run to compile @turbo/@batch paths before any timed benchmarks.
println("JIT warmup (small trial run)...")
warmup_n = minimum(n_ps)
warmup_max_pts = 2^8
warmup_opts = timed_qmc_opts(warmup_max_pts)
warmup_M = ones(warmup_n, warmup_n)
warmup_M[diagind(warmup_M)] .= 2.0
trial_run!(warmup_M, warmup_n, b0, warmup_opts)
reset_all_timers!()

df_timeit = DataFrame()

for qmc_pts in m_values
    max_pts = Int(qmc_pts ÷ 12)
    opts = timed_qmc_opts(max_pts)
    println("Running for qmc_pts = $qmc_pts (max_pts = $max_pts)")

    for n_p in n_ps
        println("  n_p = $n_p")
        local M = ones(n_p, n_p)
        M[diagind(M)] .= 2.0
        reset_all_timers!()

        for _ in 1:n_reps
            GC.enable(false)

            @timeit to "Total" begin
                @timeit to "Pre-allocation" data_i = QMCData_timeit(copy(M), -Inf * ones(n_p), b0 * ones(n_p); opts=opts)
                qmc_pnorm!(data_i, Random.default_rng())
            end

            GC.enable(true)
        end

        df_n = combined_timer_df(to, tos)
        df_n.n .= n_p
        df_n.n_pts .= qmc_pts
        append!(df_timeit, df_n)
    end
end

CSV.write(sim_resultpath("qmc_parts_timed.csv"), df_timeit)
