using LinearAlgebra, Random, SpecialFunctions



struct CholeskyGenz{T}
    U::Matrix{T}
    perm::Vector{Int}
    rank::Int
    a::Vector{T}
    b::Vector{T}
    block_size::Int
    pivot_method::Symbol
    scale_vec::Vector{T}
end



def_c_bsize(n) = n < 2^10 ? 2^5 : 2^6





function adapt_low_rank!(M::StridedMatrix{T}, a::Vector{T}, b::Vector{T}, rank::Int, perm::Vector{Int}, eps1::T) where {T<:AbstractFloat}
    n = size(M, 2)
    # k_js tracks the current end-index of the block for each rank level.
    # Initially, for the non-singular part 1:rank, block i ends at index i.
    k_js = collect(1:rank)
    k_0 = 0

    # Pre-allocate temporary storage for swapping columns
    temp_vec = zeros(T, size(M, 1))

    # Iterate over the singular/dependent variables (columns rank+1 to n)
    j = rank + 1
    n_active = n

    while j <= n_active
        # 1. Identify the level i_j (the index of the last non-zero element in column j)
        i_j = findlast(i -> abs(M[i, j]) > eps1, 1:rank)

        # If column is effectively zero, we treat it as level 0 and move it at the end
        if isnothing(i_j)
            # Move column j to n_active
            if j != n_active
                # Swap columns j and n_active
                copyto!(temp_vec, view(M, :, n_active))
                copyto!(view(M, :, n_active), view(M, :, j))
                copyto!(view(M, :, j), temp_vec)

                # Swap associated vectors
                a[j], a[n_active] = a[n_active], a[j]
                b[j], b[n_active] = b[n_active], b[j]
                perm[j], perm[n_active] = perm[n_active], perm[j]
            end
            k_0 += 1
            n_active -= 1
            # Do NOT increment j, as we need to process the column swapped into j
            continue
        end

        # 2. Scale the constraint row and bounds 
        m_ij = M[i_j, j]
        is_neg = m_ij < zero(T)
        inv_m = one(T) / m_ij

        # Scale the column j up to i_j
        for k in 1:i_j
            M[k, j] *= inv_m
        end

        # Scale bounds
        a[j] *= inv_m
        b[j] *= inv_m

        # If scaling factor was negative, swap lower and upper bounds 
        if is_neg
            a[j], b[j] = b[j], a[j]
        end

        # 3. Insert column j into the correct sorted block
        # The constraint j depends on 1..i_j. It should be grouped with level i_j.
        # k_js[i_j] points to the end of the current block for level i_j.
        # We insert the new constraint at pos = k_js[i_j] + 1.
        pos = k_js[i_j] + 1

        # If j is already at the correct position, just update counter
        if j != pos
            # Shift columns M[:, pos:j-1] to the right (to pos+1:j)
            # We copy column j to temp, shift the block right, then place temp at pos.

            # Save current column j (the constraint we are moving)
            copyto!(temp_vec, view(M, :, j))
            val_a = a[j]
            val_b = b[j]
            val_p = perm[j]

            # Shift data to the right to make space at `pos`
            # Using view for efficient memory access
            copyto!(view(M, :, pos+1:j), view(M, :, pos:j-1))
            copyto!(view(a, pos+1:j), view(a, pos:j-1))
            copyto!(view(b, pos+1:j), view(b, pos:j-1))
            copyto!(view(perm, pos+1:j), view(perm, pos:j-1))

            # Place the singular constraint at the correct sorted position
            copyto!(view(M, :, pos), temp_vec)
            a[pos] = val_a
            b[pos] = val_b
            perm[pos] = val_p
        end

        # 4. Update the block pointers
        # Since we inserted an element at level i_j, all blocks from i_j to rank
        # have shifted to the right by 1.
        for k in i_j:rank
            k_js[k] += 1
        end

        j += 1
    end

    return k_js, k_0
end




















function cholesky_classic!(M::StridedMatrix{T}, a::Vector{T}, b::Vector{T}, block_size::Int=2^7, block_size2::Int=2^9) where {T<:AbstractFloat}
    n = size(M, 1)
    eps0 = eps() * maximum(abs, view(M, diagind(M)))
    scale_vec = Vector{T}(undef, n)

    @inbounds for j in 1:n
        val = M[j, j]
        scale_vec[j] = val > 0 ? sqrt(val) : one(T)
    end

    for j in 1:n
        s_j = scale_vec[j]
        inv_s_j = 1 / s_j

        M[j, j] = one(T)
        a[j] *= inv_s_j
        b[j] *= inv_s_j

        @turbo for i in 1:(j-1)
            s_i = scale_vec[i]
            M[i, j] = M[i, j] * (inv_s_j / s_i)
        end
    end

    @inbounds for ii0 in 1:block_size2:n
        ii1 = min(n, ii0 + block_size2 - 1)

        for i0 in ii0:block_size:ii1
            i1 = min(ii1, i0 + block_size - 1)

            for i in i0:i1
                m_ii = M[i, i]
                if i > i0
                    m_ii -= dot(view(M, i0:i-1, i), view(M, i0:i-1, i))
                end
                m_ii = sqrt(m_ii)
                M[i, i] = m_ii
                if (i > i0) && (i < i1)
                    for j in (i+1):i1
                        c = dot(view(M, i0:i-1, j), view(M, i0:i-1, i))
                        M[i, j] -= c
                        M[j, i] = zero(T)
                    end
                end

                for j in (i+1):i1
                    M[i, j] /= m_ii
                end
            end

            if i1 < ii1
                ldiv!(UpperTriangular(view(M, i0:i1, i0:i1))',
                    view(M, i0:i1, (i1+1):ii1))
                mul!(view(M, (i1+1):ii1, (i1+1):ii1),
                    transpose(view(M, i0:i1, (i1+1):ii1)),
                    view(M, i0:i1, (i1+1):ii1),
                    -one(T), one(T))
            end
        end

        if ii1 < n
            ldiv!(UpperTriangular(view(M, ii0:ii1, ii0:ii1))',
                view(M, ii0:ii1, (ii1+1):n))
            mul!(view(M, (ii1+1):n, (ii1+1):n),
                transpose(view(M, ii0:ii1, (ii1+1):n)),
                view(M, ii0:ii1, (ii1+1):n),
                -one(T), one(T))
        end
    end

    return CholeskyGenz(M, collect(1:n), n, a, b, block_size, :none, scale_vec)

end























function cholesky_rowmax!(M::StridedMatrix{T}, a::Vector{T}, b::Vector{T}, block_size::Int=2^7, block_size2::Int=2^9) where {T<:AbstractFloat}
    n = size(M, 1)
    perm = collect(1:n)
    rank = n
    d_vec = zeros(T, n)

    scale_vec = ones(T, n)
    eps0 = eps() * maximum(abs, view(M, diagind(M)))

    @inbounds for j in 1:n
        val = M[j, j]
        scale_vec[j] = val > 0 ? sqrt(val) : one(T)
    end

    for j in 1:n
        s_j = scale_vec[j]
        inv_s_j = 1 / s_j

        M[j, j] = one(T)
        a[j] *= inv_s_j
        b[j] *= inv_s_j

        @turbo for i in 1:(j-1)
            s_i = scale_vec[i]
            val = M[i, j] * (inv_s_j / s_i)
            M[i, j] = val
            M[j, i] = val
        end
    end

    @inbounds for ii0 in 1:block_size2:n
        ii1 = min(n, ii0 + block_size2 - 1)

        for i0 in ii0:block_size:ii1
            i1 = min(ii1, i0 + block_size - 1)

            for k in i0:ii1
                d_vec[k] = zero(T)
            end

            for i in i0:i1
                i_m = i
                m_ii = M[i, i] - d_vec[i]

                for k in i:ii1
                    m_kk = M[k, k] - d_vec[k]

                    if m_kk > m_ii
                        m_ii = m_kk
                        i_m = k
                    end
                end



                if m_ii > 0
                    m_ii = sqrt(m_ii)
                else
                    rank = i - 1
                    break
                end

                if i_m != i
                    perm[i], perm[i_m] = perm[i_m], perm[i]
                    d_vec[i_m], d_vec[i] = d_vec[i], d_vec[i_m]


                    for k in i0:n
                        M[i, k], M[i_m, k] = M[i_m, k], M[i, k]
                    end
                    for k in 1:n
                        M[k, i], M[k, i_m] = M[k, i_m], M[k, i]
                    end
                end


                M[i, i] = m_ii


                if i == i0
                    for j in (i+1):ii1
                        m_ij = M[i, j] / m_ii
                        M[i, j] = m_ij
                        M[j, i] = m_ij
                        d_vec[j] += m_ij^2
                    end
                elseif (i < ii1)
                    mul!(
                        view(M, (i+1):ii1, i),
                        view(M, (i+1):ii1, i0:(i-1)),
                        view(M, i0:(i-1), i),
                        -one(T), one(T)
                    )

                    for j in (i+1):ii1
                        m_ij = M[j, i]
                        m_ij /= m_ii
                        M[i, j] = m_ij
                        M[j, i] = m_ij
                        d_vec[j] += m_ij^2
                    end
                end

            end

            if rank < n
                break
            end

            if i1 < ii1
                mul!(view(M, (i1+1):ii1, (i1+1):ii1),
                    view(M, (i1+1):ii1, i0:i1),
                    view(M, i0:i1, (i1+1):ii1), -one(T), one(T))
            end
        end

        if rank < n
            break
        end

        if ii1 < n
            ldiv!(UpperTriangular(view(M, ii0:ii1, ii0:ii1))',
                view(M, ii0:ii1, (ii1+1):n))
            mul!(view(M, (ii1+1):n, (ii1+1):n),
                transpose(view(M, ii0:ii1, (ii1+1):n)),
                view(M, ii0:ii1, (ii1+1):n),
                -one(T), one(T))
        end
    end

    return CholeskyGenz(M, perm, rank, a, b, block_size, :row_max, scale_vec)

end












function cholesky_genz!(M::StridedMatrix{T}, a::Vector{T}, b::Vector{T}, block_size::Int=2^7, block_size2::Int=2^9) where {T<:AbstractFloat}
    c2 = T(1.0 / sqrt(2.0))
    # Precompute factor: 2 / sqrt(2π)
    is2pi = 2.0 / sqrt(2 * π)
    n = size(M, 1)
    eps0 = eps() * maximum(abs, view(M, diagind(M)))
    scale_vec = ones(T, n)

    @inbounds for j in 1:n
        val = M[j, j]
        scale_vec[j] = val > 0 ? sqrt(val) : one(T)
    end

    for j in 1:n
        s_j = scale_vec[j]
        inv_s_j = 1 / s_j

        M[j, j] = one(T)
        a[j] *= inv_s_j
        b[j] *= inv_s_j

        # This loop is simple enough for @turbo
        @turbo for i in 1:(j-1)
            s_i = scale_vec[i]
            val = M[i, j] * (inv_s_j / s_i)
            M[i, j] = val
            M[j, i] = val
        end
    end

    perm = collect(1:n)
    rank = n
    d_vec = zeros(T, n)
    s_vec = zeros(T, n)
    order_vec = zeros(T, n)
    y_i = zero(T)

    @inbounds for ii0 in 1:block_size2:n
        ii1 = min(n, ii0 + block_size2 - 1)

        for i0 in ii0:block_size:ii1
            i1 = min(ii1, i0 + block_size - 1)

            for k in i0:ii1
                d_vec[k] = zero(T)
            end

            for i in i0:i1

                @turbo for k in i:ii1
                    m_kk = M[k, k] - d_vec[k]

                    # Ensure non-negative before sqrt (standard Cholesky safety)
                    m_kk = m_kk < 0 ? zero(T) : sqrt(m_kk)

                    a_k = (a[k] - s_vec[k]) / m_kk * c2
                    b_k = (b[k] - s_vec[k]) / m_kk * c2

                    # diff_k here is approx 2 * ProbabilityMass
                    diff_k = erf(b_k) - erf(a_k)
                    order_vec[k] = diff_k
                end

                i_m = argmin(view(order_vec, i:ii1)) + i - 1
                d_ii = order_vec[i_m]
                m_ii = M[i_m, i_m] - d_vec[i_m]

                if i_m != i
                    perm[i], perm[i_m] = perm[i_m], perm[i]
                    d_vec[i_m], d_vec[i] = d_vec[i], d_vec[i_m]
                    s_vec[i_m], s_vec[i] = s_vec[i], s_vec[i_m]
                    a[i_m], a[i] = a[i], a[i_m]
                    b[i_m], b[i] = b[i], b[i_m]

                    for k in i0:n
                        M[i, k], M[i_m, k] = M[i_m, k], M[i, k]
                    end
                    for k in 1:n
                        M[k, i], M[k, i_m] = M[k, i_m], M[k, i]
                    end
                end

                if m_ii > eps0
                    m_ii = sqrt(m_ii)
                else
                    rank = i - 1
                    break
                end

                M[i, i] = m_ii

                a_norm = (a[i] - s_vec[i]) / m_ii
                b_norm = (b[i] - s_vec[i]) / m_ii

                y_i = (exp(-a_norm^2 / 2) - exp(-b_norm^2 / 2)) * is2pi / d_ii

                if i == i0
                    for j in (i+1):ii1
                        m_ij = M[i, j] / m_ii
                        M[i, j] = m_ij
                        M[j, i] = m_ij
                        d_vec[j] += m_ij^2
                        s_vec[j] += y_i * m_ij
                    end
                elseif (i < ii1)
                    mul!(
                        view(M, (i+1):ii1, i),
                        view(M, (i+1):ii1, i0:(i-1)),
                        view(M, i0:(i-1), i),
                        -one(T), one(T)
                    )

                    for j in (i+1):ii1
                        m_ij = M[j, i]
                        m_ij /= m_ii
                        M[i, j] = m_ij
                        M[j, i] = m_ij
                        d_vec[j] += m_ij^2
                        s_vec[j] += y_i * m_ij
                    end
                end
            end

            if rank < n
                break
            end

            if i1 < ii1
                mul!(view(M, (i1+1):ii1, (i1+1):ii1),
                    view(M, (i1+1):ii1, i0:i1),
                    view(M, i0:i1, (i1+1):ii1), -one(T), one(T))
            end
        end

        if rank < n
            break
        end

        if ii1 < n
            ldiv!(UpperTriangular(view(M, ii0:ii1, ii0:ii1))',
                view(M, ii0:ii1, (ii1+1):n))
            mul!(view(M, (ii1+1):n, (ii1+1):n),
                transpose(view(M, ii0:ii1, (ii1+1):n)),
                view(M, ii0:ii1, (ii1+1):n),
                -one(T), one(T))
        end
    end

    return CholeskyGenz(M, perm, rank, a, b, block_size, :genz, scale_vec)
end