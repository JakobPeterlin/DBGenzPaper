include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie
using LaTeXStrings

function plot_benchmarks(results::DataFrame...;
    backend_names=("OpenBLAS", ACCELERATED_BLAS_FULL_LABEL),
    savepath=nothing
)
    combined = DataFrame()
    for (res, name) in zip(results, backend_names)
        df = copy(res)
        df.backend = fill(name, nrow(df))
        combined = vcat(combined, df)
    end

    name_map = Dict(
        "chol_classic" => "classic",
        "chol_rowmax" => "rowmax",
        "chol_genz" => "genz",
        "cholesky_blas" => "BLAS",
        "cholesky_blas_pivoted" => "BLAS_pivoted",
    )
    combined.chol_fun = [get(name_map, String(x), String(x)) for x in combined.chol_fun]

    cols_to_keep = [:n, :backend, :chol_fun]
    stats_cols = [:t_min, :t_median]
    df_long = stack(combined, stats_cols, cols_to_keep; variable_name=:stat, value_name=:time)
    df_long.stat = [replace(String(s), "t_" => "") for s in df_long.stat]

    ns_sorted = sort!(unique(combined.n))
    log2_ns = Int.(log2.(ns_sorted))
    tick_labels = [L"2^{%$k}" for k in log2_ns]

    plt = AlgebraOfGraphics.data(df_long) *
          AlgebraOfGraphics.mapping(
        :n, :time;
        color=:chol_fun => "Function",
        marker=:stat => "Measurement",
        col=:backend,
    ) *
          AlgebraOfGraphics.visual(ScatterLines)

    fig = draw(
        plt;
        figure=(size=(600, 500),),
        axis=(
            xticks=(ns_sorted, tick_labels),
            xlabel="Matrix size (n)",
            ylabel="Ratio of time compared to the minimum of BLAS",
            xscale=log2,
        ),
        legend=(
            position=:bottom,
            orientation=:horizontal,
            nbanks=2,
            labelsize=10,
            titleposition=:left,
        ),
    )

    if !isnothing(savepath)
        save(savepath, fig)
    end
    return fig
end

results_OB = CSV.read(sim_resultpath("results_OB.csv"), DataFrame)
results_accelerated = CSV.read(sim_resultpath("results_$(ACCELERATED_BLAS_TAG).csv"), DataFrame)

plot_benchmarks(
    results_OB,
    results_accelerated;
    savepath=resultpath("cholesky_benchmarks.pdf"),
)
