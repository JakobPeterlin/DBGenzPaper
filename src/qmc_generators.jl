

using Sobol, Primes, Random, LoopVectorization




struct RichtmyerQMC{T}
    q::Vector{Float64}
    X::Matrix{T}
    shifts::Matrix{T}
    n_base::Ref{Int}
end


struct SobolQMC{T,d}
    n_bits::Int
    m::Int
    skip0::Int


    sobol_gen::Base.RefValue{SobolSeq{d}}
    X_F::Matrix{Float64}
    X::Matrix{UInt64}
    shifts::Matrix{UInt64}
end












QMCGenrator{T} = Union{RichtmyerQMC{T},SobolQMC{T}}







function RichtmyerQMC(T, n::Int, m::Int, n_reps::Int, rng::AbstractRNG=Random.default_rng())
    q = zeros(Float64, n)
    get_sqrt_primes!(q)
    X = zeros(T, m, n)
    richtmyer_mat!(X, q, 0)
    shifts = rand(rng, T, n, n_reps)
    return RichtmyerQMC{T}(q, X, shifts, Ref(0))
end


function SobolQMC(T, d::Int, m::Int, n_reps::Int, rng; n_bits::Int=32, skip0::Int=(2 * m - 1))
    s0 = skip(SobolSeq(d), skip0)
    X_F = zeros(Float64, d, m)
    sobol_mat!(X_F, s0)
    X = zeros(UInt64, m, d)
    copyt_as_UInt64!(X, X_F, n_bits)
    shifts = rand(rng, UInt64, d, n_reps) # (d × n_reps) like qmc6.jl
    return SobolQMC{T,d}(n_bits, m, skip0, Ref(s0), X_F, X, shifts)
end




























function reset_gen!(R::RichtmyerQMC{T}, rng::AbstractRNG=Random.default_rng()) where T
    R.n_base[] = 0
    rand!(rng, R.shifts)
    richtmyer_mat!(R.X, R.q, 0)

end


function reset_gen!(S::SobolQMC{T,d}, rng::AbstractRNG=Random.default_rng()) where {T,d}
    S.sobol_gen[] = skip(SobolSeq(d), S.skip0)
    sobol_mat!(S.X_F, S.sobol_gen[])
    copyt_as_UInt64!(S.X, S.X_F, S.n_bits)

    rand!(rng, S.shifts)
end










function next_points!(R::RichtmyerQMC{T}) where T
    R.n_base[] += size(R.X, 1)
    richtmyer_mat!(R.X, R.q, R.n_base[])
end





function next_points!(S::SobolQMC{T}) where {T}
    sobol_mat!(S.X_F, S.sobol_gen[])
    copyt_as_UInt64!(S.X, S.X_F, S.n_bits) # update current block
end





@inline function rand_points!(
    Y_j::AbstractMatrix{T},
    R::RichtmyerQMC{T},
    r_j::UnitRange{Int},
    i::Int,
    k::Int,
    c::AbstractVector{T},
    dc::AbstractVector{T},
    ep0::T,
    ep1::T,
) where {T}

    shift_i = R.shifts[i, k]
    c2 = T(2.0)
    c1 = one(T)
    cm1 = -one(T)
    j1 = first(r_j)
    X = R.X

    @turbo for t in 1:length(r_j)
        j = j1 + (t - 1)
        u = (shift_i + X[j, i])
        u = u > c1 ? u - c1 : u
        u = abs(muladd(u, c2, cm1))
        u = muladd(u, dc[t], c[t])
        u = min(max(u, ep0), ep1)
        Y_j[t, i] = muladd(c2, u, cm1)
    end
end









@inline function rand_points!(
    Y_j::AbstractMatrix{T},
    S::SobolQMC{T},
    r_j::UnitRange{Int},
    i::Int,
    k::Int,
    c::AbstractVector{T},
    dc::AbstractVector{T},
    ep0::T,
    ep1::T,
) where {T}
    L = S.n_bits
    mask = (UInt64(1) << L) - 1
    invscale = ldexp(one(T), -L)
    shift_i = S.shifts[i, k] & mask
    c2 = one(T) + one(T)
    c1 = one(T)
    cm1 = -one(T)
    j1 = first(r_j)

    @turbo for t in 1:length(r_j)
        j = j1 + (t - 1)
        x_ij = S.X[j, i] & mask
        u = T((x_ij ⊻ shift_i) & mask) * invscale
        u = abs(muladd(u, c2, cm1))
        u = muladd(u, dc[t], c[t])
        u = min(max(u, ep0), ep1)
        Y_j[t, i] = muladd(c2, u, cm1)
    end
end












@inline function rand_points!(
    Y_block::AbstractMatrix{T},
    R::RichtmyerQMC{T},
    r_j::UnitRange{Int},
    i::Int,
    i_local::Int,
    k::Int,
    c::AbstractVector{T},
    dc::AbstractVector{T},
    ep0::T,
    ep1::T,
) where T
    n_pts = R.n_base[]
    c2 = one(T) + one(T)
    c1 = one(T)
    cm1 = -one(T)
    shift_i = R.shifts[i, k]


    @turbo for t in 1:length(r_j)
        j = first(r_j) + (t - 1)
        j_n = j + n_pts
        u = shift_i + R.X[j, i]
        u = u > c1 ? u - c1 : u
        u = abs(muladd(u, c2, cm1))
        u = muladd(u, dc[j], c[j])
        u = min(max(u, ep0), ep1)
        Y_block[t, i_local] = muladd(c2, u, cm1)
    end
end









@inline function rand_points!(
    Y_block::AbstractMatrix{T},
    S::SobolQMC{T},
    r_j::UnitRange{Int},
    i::Int,
    i_local::Int,
    k::Int,
    c::AbstractVector{T},
    dc::AbstractVector{T},
    ep0::T,
    ep1::T,
) where {T}
    L = S.n_bits
    mask = (UInt64(1) << L) - 1
    invscale = ldexp(one(T), -L)
    shift_i = S.shifts[i, k] & mask
    c2 = one(T) + one(T)
    c1 = one(T)
    cm1 = -one(T)

    @turbo for t in 1:length(r_j)
        j = first(r_j) + (t - 1)
        x_ij = S.X[j, i] & mask
        u = T((x_ij ⊻ shift_i) & mask) * invscale
        u = abs(muladd(u, c2, cm1))
        u = c[j] + u * dc[j]
        u = min(max(u, ep0), ep1)
        Y_block[t, i_local] = muladd(c2, u, cm1)
    end
end









## Helper functions




function get_sqrt_primes!(q::Vector{Float64}, p=1)
    n = length(q)

    for i in 1:n
        p = nextprime(p + 1)
        sqrt_p = sqrt(Float64(p))
        q[i] = sqrt_p - floor(sqrt_p)
    end

    return q
end



function richtmyer_mat!(X::Matrix{T}, q::Vector{Float64}, n_0::Int) where T

    for j in axes(X, 2)
        q_j = q[j]

        @turbo for i in axes(X, 1)
            x = Float64(n_0 + i) * q_j
            X[i, j] = x - floor(x)
        end
    end
    return X
end





function sobol_mat!(X, s)
    for j in axes(X, 2)
        Sobol.next!(s, view(X, :, j))
    end

    return X
end


function copyt_as_UInt64!(X, X_f, n_bits::Int=32)
    scale = Float64(UInt64(1) << n_bits)
    c1 = prevfloat(1.0)

    @turbo for j in axes(X, 2)
        for i in axes(X, 1)
            x_ij = X_f[j, i]
            x_ij = x_ij >= 1.0 ? c1 : x_ij
            X[i, j] = floor(UInt64, floor(x_ij * scale))
        end
    end
    return X
end
