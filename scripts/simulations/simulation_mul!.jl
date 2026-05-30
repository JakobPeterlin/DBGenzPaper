include("simulations_functions.jl")

# Load QMC code when this file is run standalone (avoid double-including in interactive sessions)
if !@isdefined(QMCData)
    include(joinpath(@__DIR__, "..", "..", "src", "qmc.jl"))
end



sim_reps = simcfg("simulation_mul!", "n_reps", 1000)

o1(m) = QMC_opts(m=m, max_pts=m, n_reps=1, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^6)

o1_OB(m) = QMC_opts(m=m, max_pts=m, n_reps=1, block_size_i=2^9, block_size_i2=2^6, block_size_j=2^6)






using OpenBLAS_jll


function compare_mul(n::Int, m::Int, n_reps::Int, o1;
    seed::Int=42,
    qmc_type::Symbol=:Richtmyer,
    mat_fun=rand_spd)

    rng = MersenneTwister(seed)
    opts = o1(m)

    # QMC / pnorm timing
    S = mat_fun(n; rng=rng)
    a = fill(-Inf, n)
    b = fill(Inf, n)
    # baseline
    data = QMCData(copy(S), copy(a), copy(b), opts, rng, qmc_type)

    # warmup (avoid compilation skewing the first measured repetition)
    qmc_pnorm!(data)

    pnorm_times = Vector{Float64}(undef, n_reps)

    b_t = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    for i in 1:n_reps
        pnorm_times[i] = @elapsed qmc_pnorm!(data)
    end

    BLAS.set_num_threads(b_t)

    # mul! timing
    A = randn(rng, m, n)
    B = randn(rng, n, n)
    U = UpperTriangular(B)
    C = zeros(eltype(A), m, n)

    mul!(C, A, B) # warmup
    mul!(C, A, U) # warmup
    mul_times = Vector{Float64}(undef, n_reps)
    mul_times2 = Vector{Float64}(undef, n_reps)
    for i in 1:n_reps
        mul_times[i] = @elapsed mul!(C, A, B)
    end

    for i in 1:n_reps
        mul_times2[i] = @elapsed mul!(C, A, U)
    end

    return (pnorm_times=pnorm_times, mul_times=mul_times, mul_times2=mul_times2)
end


function run_mul_comparissons(ns, ms, n_reps::Int, opts;
    seed::Int=42,
    qmc_type::Symbol=:Richtmyer,
    mat_fun=rand_spd)

    df = DataFrame(
        n=Int[],
        m=Int[],
        method=String[],
        min=Float64[],
        median=Float64[],
    )

    @showprogress for n in ns
        for m in ms
            # deterministic per (n,m) while still varying across the grid
            seed_nm = seed + 10_000 * n + m
            times = compare_mul(n, m, n_reps, opts; seed=seed_nm, qmc_type=qmc_type, mat_fun=mat_fun)

            push!(df, (n=n, m=m, method="pnorm", min=minimum(times.pnorm_times), median=Statistics.median(times.pnorm_times)))
            push!(df, (n=n, m=m, method="mul!", min=minimum(times.mul_times), median=Statistics.median(times.mul_times)))
            push!(df, (n=n, m=m, method="mul!_UpperTriangular", min=minimum(times.mul_times2), median=Statistics.median(times.mul_times2)))
        end
    end

    return df
end



use_accelerated_blas!()

@time mul_comp_accelerated = run_mul_comparissons(2 .^ [9, 10, 11, 12, 13], 2 .^ (10, 14), sim_reps, o1)
mul_comp_accelerated.BLAS .= ACCELERATED_BLAS_LABEL


openblas_path = OpenBLAS_jll.libopenblas_path
LinearAlgebra.BLAS.lbt_forward(openblas_path; clear=true)

@time mul_comp_OB = run_mul_comparissons(2 .^ [9, 10, 11, 12, 13], 2 .^ (10, 14), sim_reps, o1_OB)
mul_comp_OB.BLAS .= "OpenBLAS"




use_accelerated_blas!()
println(BLAS.get_config())


mul_comp = vcat(mul_comp_accelerated, mul_comp_OB)
CSV.write(sim_resultpath("mul_comp.csv"), mul_comp)










