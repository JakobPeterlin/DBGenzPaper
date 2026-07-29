include("simulations_functions.jl")

use_accelerated_blas!()

"""
    precision_qmc_opts(max_pts)

Create the options shared by the `QMCData` and `QMCData_Distributions`
calculations. These match `run_pnorm` in `simulation_comparison.jl`.
"""
function precision_qmc_opts(max_pts::Int)
    if use_MKL_instead_of_ACC
        return QMC_opts(
            Float64;
            chol_block_size=2^9,
            chol_block_size2=2^9,
            m=max_pts,
            max_pts=max_pts,
            max_abs_err=0.0,
            block_size_i=2^9,
            block_size_i2=2^6,
            block_size_j=2^6,
        )
    end

    return QMC_opts(
        Float64;
        chol_block_size=2^5,
        chol_block_size2=2^7,
        m=max_pts,
        max_pts=max_pts,
        max_abs_err=0.0,
        block_size_i=2^10,
        block_size_i2=2^6,
        block_size_j=2^7,
    )
end

"""
    simulation_precision(n_dim, n_reps, rng;
        simulation_functions=(rand_spd, mattern_cov1),
        max_pts=2^10, qmc_type=:Richtmyer)

    simulation_precision(; n_dim=2 .^ [6, 8, 10, 12],
        n_reps=simcfg("simulation_mul!", "n_reps", 1000),
        rng=MersenneTwister(42), simulation_functions=(rand_spd, mattern_cov1),
        max_pts=2^10, qmc_type=:Richtmyer)

Compare `qmc_pnorm!` using `QMCData` with the implementation using
`QMCData_Distributions`. A new matrix and matching bounds are generated for
every repetition. The construction of the two data objects is deliberately
outside the timed regions.

The two QMC generators receive independent RNGs initialized with the same
seed, so each pair of calls uses identical scrambled points.
"""
function simulation_precision(
    n_dim::AbstractVector{<:Integer},
    n_reps::Integer,
    rng::AbstractRNG;
    simulation_functions=(rand_spd, mattern_cov1),
    max_pts::Integer=2^10,
    qmc_type::Symbol=:Richtmyer,
)
    n_reps > 0 || throw(ArgumentError("n_reps must be positive"))
    max_pts > 0 || throw(ArgumentError("max_pts must be positive"))
    all(>(1), n_dim) || throw(ArgumentError("all dimensions must be greater than one"))

    opts = precision_qmc_opts(Int(max_pts))
    results = DataFrame(
        simulation_function=String[],
        n=Int[],
        rep=Int[],
        qmc_data_value=Float64[],
        qmc_data_distributions_value=Float64[],
        absolute_difference=Float64[],
        qmc_data_time=Float64[],
        qmc_data_distributions_time=Float64[],
    )

    @showprogress for simulation_function in simulation_functions
        function_name = string(nameof(simulation_function))

        for n_raw in n_dim
            n = Int(n_raw)
            println("n_dim = ", n)

            for rep in 1:n_reps
                Σ = simulation_function(n; rng=rng)
                k = quantile(Normal(), (1 + 0.25^(1 / n)) / 2)
                a = -k * sqrt.(diag(Σ))
                b = k * sqrt.(diag(Σ))

                # Draw one seed per repetition and use it twice. This advances
                # the caller's RNG while keeping the paired scrambles equal.
                qmc_seed = rand(rng, UInt64)
                qmc_rng = MersenneTwister(qmc_seed)
                distributions_rng = MersenneTwister(qmc_seed)

                qmc_data = QMCData(
                    copy(Σ),
                    copy(a),
                    copy(b),
                    opts,
                    qmc_rng,
                    qmc_type,
                )
                distributions_data = DBGenzPaper.QMCData_Distributions(
                    QMCData(
                        copy(Σ),
                        copy(a),
                        copy(b),
                        opts,
                        distributions_rng,
                        qmc_type,
                    ),
                )

                qmc_result = nothing
                distributions_result = nothing
                qmc_time = @elapsed qmc_result = qmc_pnorm!(qmc_data)
                distributions_time =
                    @elapsed distributions_result = qmc_pnorm!(distributions_data)

                qmc_value = Float64(qmc_result[1])
                distributions_value = Float64(distributions_result[1])
                push!(
                    results,
                    (
                        simulation_function=function_name,
                        n=n,
                        rep=rep,
                        qmc_data_value=qmc_value,
                        qmc_data_distributions_value=distributions_value,
                        absolute_difference=abs(qmc_value - distributions_value),
                        qmc_data_time=qmc_time,
                        qmc_data_distributions_time=distributions_time,
                    ),
                )
            end
        end
    end

    return results
end

function simulation_precision(
    n_dim::AbstractVector{<:Integer},
    n_reps::Integer,
    rng::AbstractRNG,
    simulation_functions;
    kwargs...,
)
    return simulation_precision(
        n_dim,
        n_reps,
        rng;
        simulation_functions=simulation_functions,
        kwargs...,
    )
end

function simulation_precision(;
    n_dim::AbstractVector{<:Integer}=2 .^ [6, 8, 10, 12],
    n_reps::Integer=Int(simcfg("simulation_mul!", "n_reps", 1000)),
    rng::AbstractRNG=MersenneTwister(42),
    simulation_functions=(rand_spd, mattern_cov1),
    max_pts::Integer=2^11,
    qmc_type::Symbol=:Richtmyer,
)
    return simulation_precision(
        n_dim,
        n_reps,
        rng;
        simulation_functions=simulation_functions,
        max_pts=max_pts,
        qmc_type=qmc_type,
    )
end

precision_n_dim =
    2 .^ [Int(x) for x in simcfg("simulation_precision", "n_powers", [6, 8, 10, 12])]
precision_n_reps = Int(simcfg("simulation_mul!", "n_reps", 1000))
precision_max_pts = Int(simcfg("simulation_precision", "max_pts", 2^10))
precision_seed = Int(simcfg("simulation_precision", "seed", 42))

precision_results = simulation_precision(
    precision_n_dim,
    precision_n_reps,
    MersenneTwister(precision_seed);
    max_pts=precision_max_pts,
)

CSV.write(sim_resultpath("precision.csv"), precision_results)
