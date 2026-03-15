include(joinpath(@__DIR__, "setup.jl"))

using Sobol, Random, BenchmarkTools, LinearAlgebra, LoopVectorization, Polyester, AppleAccelerate, TimerOutputs



const to = TimerOutput()
const tos = [TimerOutput() for i in 1:Threads.nthreads()]

include(joinpath(@__DIR__, "..", "..", "src", "cholesky.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "qmc_generators.jl"))



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


struct QMCData_timeit{T}
    C::CholeskyGenz{T}
    a::Vector{T}
    b::Vector{T}
    qmc_generator::QMCGenrator{T}
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
    m::Integer=10^4,
    block_size_i=2^6,
    block_size_j::Int=2^6,
    n_blocks=Threads.nthreads(),
    n_reps::Int=12,
    n_bits::Int=32,
    max_pts::Int=10^4,
    max_abs_err=10^(-12))

    return QMCOpts{T}(m, block_size_i, block_size_j, n_blocks, n_reps, n_bits, max_pts, max_abs_err)
end




function QMCData_timeit(C::Matrix{T0},
    a::Vector{T0},
    b::Vector{T0};
    opts::QMCOpts{T}=QMC_opts(T),
    rng=Random.default_rng(),
    qmc_type=:Richtmyer) where {T,T0}

    m = opts.m
    n_threads = Threads.nthreads()
    n_blocks = max(opts.n_blocks, n_threads)
    block_size_j = opts.block_size_j
    block_size_i = opts.block_size_i
    n_reps = opts.n_reps
    n_bits = opts.n_bits
    n = size(C, 1)

    if (n - 1) < block_size_i
        block_size_i = n - 1
    end

    opts_use = (n_blocks == opts.n_blocks && block_size_i == opts.block_size_i) ?
               opts :
               QMCOpts{T}(m, block_size_i, block_size_j, n_blocks, n_reps, n_bits, opts.max_pts, opts.max_abs_err)

    qmc_generator = @timeit to "Generating QMC points" begin
        if qmc_type == :Sobol
            SobolQMC(T, n - 1, m, n_reps, rng; n_bits=n_bits, skip0=(2 * m - 1))
        else
            RichtmyerQMC(T, n - 1, m, n_reps, rng)
        end
    end

    Ys = [zeros(T, block_size_j, n - 1) for _ in 1:n_blocks]
    sub_mats = [zeros(T, block_size_j, block_size_i) for _ in 1:n_blocks]
    c_vecs = [zeros(T, block_size_j) for _ in 1:n_blocks]
    dc_vecs = deepcopy(c_vecs)
    p_vecs = deepcopy(c_vecs)
    qmc_reps = zeros(T, n_reps)
    sum_p_threads = zeros(T, n_blocks)
    @timeit to "Cholesky" chol = T0 == T ? cholesky_genz!(copy(C), copy(a), copy(b)) : cholesky_genz!(convert.(T, C), convert.(T, a), convert.(T, b))

    if chol.rank == n
        return QMCData_timeit{T}(chol, chol.a, chol.b, qmc_generator, Ys, sub_mats, c_vecs, dc_vecs, p_vecs, qmc_reps, sum_p_threads, opts_use)
    else
        throw("Cholesky decomposition is not full rank")
    end
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











@inline function qmc_loop!(D::QMCData_timeit{T}, n_pts0::Int, c_1::T, dc_1::T) where T
    n = length(D.b)
    opts = D.qmc_opts
    gen = D.qmc_generator
    ep0 = eps(T)
    ep1 = one(T) - eps(T)
    c05 = T(0.5)
    c2 = T(sqrt(2.0))
    c_min = T(9.0)
    c_max = T(-9.0)

    @inbounds for k in eachindex(D.qmc_reps)
        sum_p_threads = D.sum_p_threads
        fill!(sum_p_threads, zero(T))

        Threads.@threads for j1 in 1:opts.block_size_j:opts.m
            j2 = min(j1 + opts.block_size_j - 1, opts.m)
            r_j = j1:j2
            len_j = length(r_j)
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

            for i1 in 1:opts.block_size_i:(n-1)
                i2 = min(i1 + opts.block_size_i - 1, n - 1)
                iblock_len = i2 - i1 + 1

                @timeit tos[i_t] "BLAS mul!" if i1 > 1
                    mul!(view(s_j, 1:len_j, 1:iblock_len),
                        view(Y_j, 1:len_j, 1:(i1-1)),
                        view(D.C.U, 1:(i1-1), (i1+1):(i2+1)))
                end

                for i in i1:i2
                    @timeit tos[i_t] "Affine scrambling" rand_points!(Y_j, gen, r_j, i, k, c_j, dc_j, ep0, ep1)
                    @timeit tos[i_t] "Computation of quantiles" @simd for i_b in 1:len_j
                        Y_j[i_b, i] = c2 * erfinv(Y_j[i_b, i])
                    end

                    @timeit tos[i_t] "Internal multiplication" mul!(view(s_j, 1:len_j, i - i1 + 1),
                        view(Y_j, 1:len_j, i1:i),
                        view(D.C.U, i1:i, i + 1),
                        one(T), one(T))

                    a_i = D.a[i+1]
                    b_i = D.b[i+1]
                    inv_u_ii2 = one(T) / (D.C.U[i+1, i+1] * c2)

                    @timeit tos[i_t] "Clculation of CDFs" @turbo for i_b in 1:len_j
                        s_val = s_j[i_b, i-i1+1]

                        c_val = (a_i - s_val) * inv_u_ii2
                        dc_val = (b_i - s_val) * inv_u_ii2

                        c_val = erf(max(min(c_val, c_min), c_max))
                        dc_val = erf(max(min(dc_val, c_min), c_max))

                        c_val = c05 * (one(T) + c_val)
                        dc_val = c05 * (one(T) + dc_val) - c_val

                        c_j[i_b] = c_val
                        dc_j[i_b] = dc_val
                        p_j[i_b] *= dc_val
                    end
                end
            end

            sum_p_threads[i_t] += sum(view(p_j, 1:len_j))
        end

        n_pts_local = opts.m
        n_pts_total = n_pts0 + n_pts_local
        mean_rep = sum(sum_p_threads) / n_pts_total
        D.qmc_reps[k] = D.qmc_reps[k] * (n_pts0 / n_pts_total) + mean_rep
    end
end




































## Integration

function qmc_pnorm!(D::Union{QMCData_timeit{T}}, rng) where T
    n = length(D.b)
    gen = D.qmc_generator

    @timeit to "Generating QMC points" reset_gen!(gen, rng)

    fill!(D.qmc_reps, zero(T))
    n_reps = length(D.qmc_reps)



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
        return result, err_acc, n_pts
    end




    while ((n_pts < max_pts) & (err_acc > max_abs_err))
        @timeit to "Generating QMC points" next_points!(gen)

        @timeit to "DB loop" qmc_loop!(D, n_pts, c_1, dc_1)

        result = sum(D.qmc_reps) / n_reps
        err_acc = 3 * sd(D.qmc_reps, result)
        n_pts += D.qmc_opts.m
    end

    return result, err_acc, n_pts
end















## Timeroutputs helper function

using DataFrames, TimerOutputs

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
    # Outer timer (optionally drop wrapper sections to avoid double counting)
    df_to = to_df(to, 0; subtract_cholesky_from_prealloc=true)
    if !isempty(drop_sections)
        keep = .!in.(df_to.Section, Ref(drop_sections))
        df_to = df_to[keep, :]
    end

    # Merge all per-thread timers, then average their times/allocations
    df_tos = to_df(merge_timers(tos), 0; subtract_cholesky_from_prealloc=false)
    n_tos = length(tos)
    if n_tos > 0
        df_tos.Time_sec ./= n_tos
        df_tos.Alloc_MiB ./= n_tos
    end

    # Combine and compute Percent from combined sum
    df = vcat(df_to, df_tos)
    total = sum(df.Time_sec)
    df.Percent = total == 0 ? zero.(df.Time_sec) : (df.Time_sec ./ total)
    return df
end

##

df_timeit = DataFrame()
df_timeit2 = DataFrame()
nn_reps = 10
n_ps = [2^4, 2^6, 2^8, 2^10, 2^12]
n_reps = simcfg("simulation_timed", "n_reps", 10)
b0 = Float64(simcfg("simulation_timed", "b0", 3.0))

opts = QMC_opts(m=2^10, ; max_pts=2^10, block_size_j=2^6, block_size_i=2^7)
opts2 = QMC_opts(m=2^10, ; max_pts=2^14, block_size_j=2^6, block_size_i=2^8)






for n_p in n_ps
    println("Running for n_p = $n_p")
    local M = ones(n_p, n_p)
    M[diagind(M)] .= 2.0
    data_i = QMCData_timeit(copy(M), -Inf * ones(n_p), b0 * ones(n_p); opts=opts)
    qmc_pnorm!(data_i, Random.default_rng())

    reset_timer!(to)

    for to_i in tos
        reset_timer!(to_i)
    end

    for i in 1:n_reps


        GC.enable(false)

        @timeit to "Total" begin
            @timeit to "Pre-allocation" data_i = QMCData_timeit(copy(M), -Inf * ones(n_p), b0 * ones(n_p); opts=opts)
            qmc_pnorm!(data_i, Random.default_rng())
        end

        GC.enable(true)



    end

    df_n = combined_timer_df(to, tos)
    df_n.n .= n_p
    append!(df_timeit, df_n)


end









for n_p in n_ps
    println("Running for n_p = $n_p")
    local M = ones(n_p, n_p)
    M[diagind(M)] .= 2.0
    data_i = QMCData_timeit(copy(M), -Inf * ones(n_p), b0 * ones(n_p); opts=opts2)
    qmc_pnorm!(data_i, Random.default_rng())

    reset_timer!(to)

    for to_i in tos
        reset_timer!(to_i)
    end

    for i in 1:n_reps
        GC.enable(false)

        @timeit to "Total" begin
            @timeit to "Pre-allocation" data_i = QMCData_timeit(copy(M), -Inf * ones(n_p), b0 * ones(n_p); opts=opts2)
            qmc_pnorm!(data_i, Random.default_rng())
        end

        GC.enable(true)


    end

    df_n = combined_timer_df(to, tos)
    df_n.n .= n_p
    append!(df_timeit2, df_n)


end




df_timeit.n_pts .= 2^10
df_timeit2.n_pts .= 2^14

df_timeit = vcat(df_timeit, df_timeit2)





CSV.write(resultpath("qmc_parts_timed.csv"), df_timeit)

##








