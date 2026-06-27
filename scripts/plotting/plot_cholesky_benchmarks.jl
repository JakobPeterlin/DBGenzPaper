include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie
using LaTeXStrings
using CategoricalArrays

const CHOLESKY_MACHINE_ORDER = ["Apple M2 Ultra", "Intel Xeon"]
const CHOLESKY_BLAS_ORDER = ["OpenBLAS", "Accelerated BLAS"]
const CHOLESKY_BLAS_METHODS = Set(["BLAS", "BLAS_pivoted"])
const CHOLESKY_METHOD_FAMILY_ORDER = ["DB", "BLAS"]
const CHOLESKY_FAMILY_COLORS = (DB=RGBf(0, 0, 1), BLAS=RGBAf(1, 0, 0, 0.8))
const CHOLESKY_POINT_COLOR_ORDER = ["classic", "rowmax", "genz", "BLAS", "BLAS_pivoted"]
const CHOLESKY_POINT_COLORS = Dict(
    "classic" => RGBf(0.0, 0.55, 0.0),
    "rowmax" => RGBf(0.9, 0.55, 0.0),
    "genz" => RGBf(0.55, 0.25, 0.75),
    "BLAS" => RGBf(0.0, 0.65, 0.75),
    "BLAS_pivoted" => RGBf(0.85, 0.45, 0.65),
)

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

    facet_mapping = AlgebraOfGraphics.mapping(
        :n,
        :time;
        group=:chol_fun,
        col=:machine => "System",
        row=:BLAS,
    )

    df_blas = df_long[in.(String.(df_long.chol_fun), Ref(CHOLESKY_BLAS_METHODS)), :]
    df_db = df_long[.!in.(String.(df_long.chol_fun), Ref(CHOLESKY_BLAS_METHODS)), :]

    line_plt = AlgebraOfGraphics.data(df_db) *
               facet_mapping *
               AlgebraOfGraphics.visual(Lines; color=CHOLESKY_FAMILY_COLORS.DB) +
               AlgebraOfGraphics.data(df_blas) *
               facet_mapping *
               AlgebraOfGraphics.visual(Lines; color=CHOLESKY_FAMILY_COLORS.BLAS)

    scatter_layers = [
        AlgebraOfGraphics.data(df_long[df_long.chol_fun .== method, :]) *
        facet_mapping *
        AlgebraOfGraphics.visual(Scatter; color=CHOLESKY_POINT_COLORS[method])
        for method in CHOLESKY_POINT_COLOR_ORDER
        if any(df_long.chol_fun .== method)
    ]

    plt = line_plt + reduce(+, scatter_layers)

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
    )

    point_methods = [
        method for method in CHOLESKY_POINT_COLOR_ORDER
        if any(df_long.chol_fun .== method)
    ]
    Legend(
        fig.figure[end + 1, :],
        [
            [
                LineElement(color=CHOLESKY_FAMILY_COLORS.DB),
                LineElement(color=CHOLESKY_FAMILY_COLORS.BLAS),
            ],
            [
                MarkerElement(color=CHOLESKY_POINT_COLORS[method], marker=:circle)
                for method in point_methods
            ],
        ],
        [CHOLESKY_METHOD_FAMILY_ORDER, point_methods],
        ["Line color", "Point color"];
        orientation=:horizontal,
        nbanks=2,
        labelsize=10,
        titleposition=:left,
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
