
include("simulations_functions.jl")


function cholesky_blas(Σ)
    Σ_copy = copy(Σ)
    return @elapsed cholesky(Σ_copy)
end

function cholesky_blas_pivoted(Σ)
    Σ_copy = copy(Σ)
    return @elapsed cholesky(Σ_copy, RowMaximum())
end
const size1c = use_MKL_instead_of_ACC ? 2^8 : 2^5
const size1 = use_MKL_instead_of_ACC ? 2^9 : 2^5
const size2 = use_MKL_instead_of_ACC ? 2^9 : 2^9






function chol_classic(Σ)
    Σ_copy = copy(Σ)
    a = -ones(size(Σ, 1))
    b = ones(size(Σ, 1))
    return @elapsed cholesky_classic!(Σ_copy, a, b, size1c, size2)
end

function chol_genz(Σ)
    Σ_copy = copy(Σ)
    a = -ones(size(Σ, 1))

    b = ones(size(Σ, 1))
    return @elapsed cholesky_genz!(Σ_copy, a, b, size1, size2)
end

function chol_rowmax(Σ)
    Σ_copy = copy(Σ)
    a = -ones(size(Σ, 1))
    b = ones(size(Σ, 1))
    return @elapsed cholesky_rowmax!(Σ_copy, a, b, size1, size2)
end





function compare_times(n::Int, n_reps::Int, chol_funs)
    times = zeros(n_reps, length(chol_funs))

    for i in 1:n_reps
        Σ = rand_spd(n; jitter=1.0)

        for j in 1:length(chol_funs)
            times[i, j] = chol_funs[j](Σ)
        end
    end

    return times
end


function benchmark_cholesky_functions(ns::Vector{Int}, n_reps::Int; chol_funs=[cholesky_blas, cholesky_blas_pivoted, chol_classic, chol_genz, chol_rowmax], standardize::Bool=true)
    """
    Iterate over a vector of n values and benchmark cholesky functions.

    Returns a DataFrame with columns: n, chol_fun, t_min, t_median, t_mean, t_max

    If standardize=true, all times are standardized by dividing by the minimum time 
    of the first function in chol_funs for each n value.
    """
    results = []

    # Get function names for labeling
    function_names = [string(nameof(f)) for f in chol_funs]

    @showprogress for n in ns
        times_matrix = compare_times(n, n_reps, chol_funs)

        # Standardize times if requested
        if standardize
            min_time_first_fun = minimum(times_matrix[:, 1])
            times_matrix = times_matrix ./ min_time_first_fun
        end

        for (idx, chol_fun) in enumerate(chol_funs)
            times = times_matrix[:, idx]

            push!(results, (
                n=n,
                chol_fun=function_names[idx],
                t_min=minimum(times),
                t_median=median(times),
                t_mean=mean(times),
                t_max=maximum(times)
            ))
        end
    end

    return DataFrame(results)
end




##
ns = [2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10, 2^11, 2^12]
n_reps = simcfg("simulation_cholesky", "n_reps", 1000)


##


openblas_path = OpenBLAS_jll.libopenblas_path
LinearAlgebra.BLAS.lbt_forward(openblas_path; clear=true)
println(BLAS.get_config())





results_OB = benchmark_cholesky_functions(ns, n_reps)
CSV.write(sim_resultpath("results_OB.csv"), results_OB)



use_accelerated_blas!()
println(BLAS.get_config())

results_accelerated = benchmark_cholesky_functions(ns, n_reps)
CSV.write(sim_resultpath("results_$(ACCELERATED_BLAS_TAG).csv"), results_accelerated)





<