include(joinpath(@__DIR__, "simulations_functions.jl"))

using Distributions


function compare_richtmyer(n::Int, m_s::Vector{Int}, n_reps::Int, mat_fun=rand_spd, rng=Random.default_rng())
    n_m = length(m_s)
    df = DataFrame(n=zeros(n_m), m=zeros(n_m), val_s=zeros(n_m), val_r=zeros(n_m), sd_s=zeros(n_m), sd_r=zeros(n_m),
        t_s=zeros(n_m), t_r=zeros(n_m))

    @showprogress for i_m in 1:length(m_s)
        m = m_s[i_m]
        opts = QMC_opts(m=m, max_pts=m, block_size_j=2^6, block_size_i=2^6)
        results_m = zeros(n_reps, 2)
        times_m = zeros(n_reps, 2)

        M = mat_fun(n; rng=rng)
        k = quantile(Normal(), (1 + 0.25^(1 / n)) / 2)
        a = -k * sqrt.(diag(M))
        b = k * sqrt.(diag(M))

        for i in 1:n_reps

            data = QMCData(copy(M), copy(a), copy(b), opts, rng, :Sobol)
            data_r = QMCData(copy(M), copy(a), copy(b), opts, rng, :Richtmyer)

            times_m[i, 1] = @elapsed results_m[i, 1] = qmc_pnorm!(deepcopy(data))[1]
            times_m[i, 2] = @elapsed results_m[i, 2] = qmc_pnorm!(deepcopy(data_r))[1]
        end

        df[i_m, :] = [n, m, mean(results_m[:, 1]), mean(results_m[:, 2]), sd(results_m[:, 1]), sd(results_m[:, 2]), median(times_m[:, 1]), median(times_m[:, 2])]
    end

    println("Finished simulations for n = $n ")
    return df
end







Random.seed!(102)
m_s = [2^10, 2^11, 2^12, 2^13, 2^14, 2^15]
n_reps = 10^3

results_r1 = compare_richtmyer(2^4, m_s, n_reps, rand_spd, Random.default_rng())
results_r2 = compare_richtmyer(2^6, m_s, n_reps, rand_spd, Random.default_rng())
results_r3 = compare_richtmyer(2^8, m_s, n_reps, rand_spd, Random.default_rng())
@time results_r4 = compare_richtmyer(2^10, m_s, n_reps, rand_spd, Random.default_rng())

results_r = vcat(results_r1, results_r2, results_r3, results_r4)
CSV.write(resultpath("results_r.csv"), results_r)

##




results_f1 = compare_richtmyer(2^4, m_s, n_reps, fixed_dense, Random.default_rng())
results_f2 = compare_richtmyer(2^6, m_s, n_reps, fixed_dense, Random.default_rng())
results_f3 = compare_richtmyer(2^8, m_s, n_reps, fixed_dense, Random.default_rng())
results_f4 = compare_richtmyer(2^10, m_s, n_reps, fixed_dense, Random.default_rng())

results_f = vcat(results_f1, results_f2, results_f3, results_f4)
CSV.write(resultpath("results_f.csv"), results_f)


##

##



results_r.Matrix .= "Random"
results_f.Matrix .= "Fixed"
