include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie
using LaTeXStrings
using CategoricalArrays

const CHOLESKY_MACHINE_ORDER = ["Apple M2 Ultra", "Intel Xeon"]
const CHOLESKY_BLAS_ORDER = ["OpenBLAS", "Accelerated BLAS"]

function read_cholesky_result(filename::AbstractString, machine::AbstractString, blas::AbstractString)
    df = CSV.read(resultpath(filename), DataFrame)
    df.machine .= machine
    df.BLAS .= blas
    return df
end

function read_cholesky_benchmarks()
    combined = vcat(
        read_cholesky_result("results_OB.csv", CHOLESKY_MACHINE_ORDER[1], "OpenBLAS"),
        read_cholesky_result("results_ACC.csv", CHOLESKY_MACHINE_ORDER[1], "Accelerated BLAS"),
        read_cholesky_result("results_OBIntel.csv", CHOLESKY_MACHINE_ORDER[2], "OpenBLAS"),
        read_cholesky_result("results_MKLIntel.csv", CHOLESKY_MACHINE_ORDER[2], "Accelerated BLAS");
        cols=:union,
    )
    combined.machine = categorical(combined.machine; ordered=true, levels=CHOLESKY_MACHINE_ORDER)
    combined.BLAS = categorical(combined.BLAS; ordered=true, levels=CHOLESKY_BLAS_ORDER)
    return combined
end

function power2_ticks(values)
    positive_values = filter(x -> isfinite(x) && x > 0, collect(skipmissing(values)))
    exps = floor(Int, log2(minimum(positive_values))):ceil(Int, log2(maximum(positive_values)))
    tick_values = 2.0 .^ exps
    tick_labels = [L"2^{%$k}" for k in exps]
    return tick_values, tick_labels
end

function plot_benchmarks(results::DataFrame; savepath=nothing)
    combined = copy(results)
    name_map = Dict(
        "chol_classic" => "classic",
        "chol_rowmax" => "rowmax",
        "chol_genz" => "genz",
        "cholesky_blas" => "BLAS",
        "cholesky_blas_pivoted" => "BLAS_pivoted",
    )
    combined.chol_fun = [get(name_map, String(x), String(x)) for x in combined.chol_fun]

    df_long = select(combined, :machine, :n, :BLAS, :chol_fun, :t_min => :time)

    ns_sorted = sort!(unique(combined.n))
    log2_ns = Int.(log2.(ns_sorted))
    tick_labels = [L"2^{%$k}" for k in log2_ns]
    y_ticks = power2_ticks(df_long.time)

    plt = AlgebraOfGraphics.data(df_long) *
          AlgebraOfGraphics.mapping(
              :n, :time;
              color=:chol_fun => "Function",
              group=:chol_fun,
              col=:machine => "System", 
              row=:BLAS,
          ) *
          (AlgebraOfGraphics.visual(Lines) + AlgebraOfGraphics.visual(Scatter))

    fig = draw(
        plt;
        figure=(size=(900, 800),),
        axis=(
            xticks=(ns_sorted, tick_labels),
            xlabel="Matrix size (n)",
            ylabel="Minimum time ratio compared to BLAS",
            xscale=log2,
            yscale=log2,
            yticks=y_ticks,
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

results = read_cholesky_benchmarks()

plot_benchmarks(
    results;
    savepath=resultpath("cholesky_benchmarks.pdf"),
)
