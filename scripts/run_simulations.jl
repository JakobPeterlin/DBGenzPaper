using Dates
using JSON

include(joinpath(@__DIR__, "simulations", "setup.jl"))

const SCRIPT_MAP = Dict(
    "simulation_comparison" => "simulations/simulation_comparison.jl",
    "simulation_sparse" => "simulations/simulation_sparse.jl",
    "simulation_just_sparse" => "simulations/simulation_just_sparse.jl",
    "simulation_FP32" => "simulations/simulation_FP32.jl",
    "simulation_precision" => "simulations/simulation_precision.jl",
    "simulation_cholesky" => "simulations/simulation_cholesky.jl",
    "simulation_mul!" => "simulations/simulation_mul!.jl",
    "simulation_richtmyer" => "simulations/simulation_richtmyer.jl",
    "simulation_timed" => "simulations/simulation_timed.jl",
    "plot_cholesky_benchmarks" => "plotting/plot_cholesky_benchmarks.jl",
    "plot_mul_benchmarks" => "plotting/plot_mul_benchmarks.jl",
    "plot_richtmyer" => "plotting/plot_richtmyer.jl",
    "plot_comparisons" => "plotting/plot_comparisons.jl",
    "plot_timed_parts" => "plotting/plot_timed_parts.jl",
    "plot_example2" => "plotting/plot_example2.jl",
    "export_tables" => "plotting/export_tables.jl",
    "example" => "example/example.jl",
    "example1" => "example/example1.jl",
    "example2" => "example/example2.jl",
)

function _run_target!(target::String)
    script_name = get(SCRIPT_MAP, target, "")
    isempty(script_name) && error("Unknown target: $(target)")
    script_path = joinpath(PROJECT_ROOT, "scripts", script_name)
    isfile(script_path) || error("Script not found: $(script_path)")

    t = @elapsed include(script_path)
    return Dict(
        "target" => target,
        "script" => script_name,
        "elapsed_seconds" => t,
    )
end

function _default_targets()
    [string(x) for x in simcfg("runner", "targets", ["simulation_comparison", "simulation_sparse"])]
end

function main(args=ARGS)
    # Run tests first to warm up JIT and fail fast on regressions.
    println("Running tests...")
    test_time = @elapsed include(joinpath(PROJECT_ROOT, "test", "runtests.jl"))
    println("Tests completed in $(round(test_time, digits=2)) seconds.")

    targets = isempty(args) ? _default_targets() : collect(args)

    println("Running targets: ", join(targets, ", "))
    results = Dict{String,Any}[]
    for target in targets
        println("-> ", target)
        push!(results, _run_target!(target))
    end

    run_manifest = Dict(
        "project" => "DBGenzPaper",
        "timestamp_utc" => string(Dates.now(Dates.UTC)),
        "config_path" => SIM_CONFIG_PATH,
        "targets" => targets,
        "results" => results,
    )

    manifest_path = resultpath("run_manifest.json")
    open(manifest_path, "w") do io
        JSON.print(io, run_manifest, 2)
    end

    println("Wrote run manifest to: ", manifest_path)
end

main()
