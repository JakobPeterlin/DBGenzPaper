include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie
using LaTeXStrings
using CategoricalArrays

const MUL_MACHINE_ORDER = ["Apple M2 Ultra", "Intel Xeon"]
const MUL_BLAS_ORDER = ["OpenBLAS", "Accelerated BLAS"]
const MUL_METHOD_ORDER = ["DB Loop", "mul!", "mul!_UpperTriangular"]
const MUL_DB_METHODS = Set(["DB Loop"])
const MUL_LINE_LABELS = ["DB", "BLAS"]
const MUL_LINE_COLORS = (DB=RGBf(0, 0, 1), BLAS=RGBAf(1, 0, 0, 0.8))
const MUL_POINT_COLORS = Dict(
    "DB Loop" => RGBf(0.0, 0.55, 0.0),
    "mul!" => RGBf(0.9, 0.55, 0.0),
    "mul!_UpperTriangular" => RGBf(0.55, 0.25, 0.75),
)
const MUL_M_SMALL = 2^10 * 7
const MUL_M_LARGE = 2^14 * 7

function read_mul_benchmarks()
    df_apple = CSV.read(resultpath("mul_comp.csv"), DataFrame)
    df_apple.machine .= MUL_MACHINE_ORDER[1]

    df_intel = CSV.read(resultpath("mul_compIntel.csv"), DataFrame)
    df_intel.machine .= MUL_MACHINE_ORDER[2]

    df = vcat(df_apple, df_intel; cols=:union)
    df.BLAS = replace(df.BLAS, "Accelerate" => "Accelerated BLAS", "MKL" => "Accelerated BLAS")
    df.machine = categorical(df.machine; ordered=true, levels=MUL_MACHINE_ORDER)
    df.BLAS = categorical(df.BLAS; ordered=true, levels=MUL_BLAS_ORDER)
    return df
end

function power2_ticks(values)
    positive_values = filter(x -> isfinite(x) && x > 0, collect(skipmissing(values)))
    exps = floor(Int, log2(minimum(positive_values))):ceil(Int, log2(maximum(positive_values)))
    tick_values = 2.0 .^ exps
    tick_labels = [L"2^{%$k}" for k in exps]
    return tick_values, tick_labels
end

function plot_mul_benchmarks(mul_comp::DataFrame, m::Int; savepath=nothing)
    df = copy(mul_comp)
    df = df[df.m.==m, :]

    baseline = combine(
        groupby(df[df.method.=="mul!", :], [:machine, :n, :m, :BLAS]),
        :min => minimum => :mul_min_min,
    )
    df = leftjoin(df, baseline, on=[:machine, :n, :m, :BLAS])
    df = df[.!ismissing.(df.mul_min_min), :]
    df.ratio = df.min ./ df.mul_min_min

    df_long = combine(
        groupby(df, [:machine, :n, :BLAS, :method]),
        :ratio => minimum => :time,
    )

    df_long.method = replace(df_long.method, "pnorm" => "DB Loop")
    df_long.method = categorical(df_long.method; ordered=true, levels=MUL_METHOD_ORDER)

    ns_sorted = sort!(unique(df_long.n))
    log2_ns = Int.(round.(log2.(ns_sorted)))
    tick_labels = [L"2^{%$k}" for k in log2_ns]
    y_ticks = power2_ticks(df_long.time)

    facet_mapping = AlgebraOfGraphics.mapping(
        :n, :time;
        group=:method,
        col=:machine => "System",
        row=:BLAS,
    )

    df_db = df_long[in.(String.(df_long.method), Ref(MUL_DB_METHODS)), :]
    df_blas = df_long[.!in.(String.(df_long.method), Ref(MUL_DB_METHODS)), :]

    line_plt = AlgebraOfGraphics.data(df_db) *
               facet_mapping *
               AlgebraOfGraphics.visual(Lines; color=MUL_LINE_COLORS.DB) +
               AlgebraOfGraphics.data(df_blas) *
               facet_mapping *
               AlgebraOfGraphics.visual(Lines; color=MUL_LINE_COLORS.BLAS)

    scatter_layers = [
        AlgebraOfGraphics.data(df_long[df_long.method.==method, :]) *
        facet_mapping *
        AlgebraOfGraphics.visual(
            Scatter;
            color=MUL_POINT_COLORS[method],
            marker=:circle,
        )
        for method in MUL_METHOD_ORDER
        if any(df_long.method .== method)
    ]

    plt = line_plt + reduce(+, scatter_layers)

    fig = draw(
        plt;
        figure=(size=(900, 800),),
        axis=(
            xticks=(ns_sorted, tick_labels),
            xlabel="Matrix size (n)",
            ylabel="Minimum time ratio (method / mul!)",
            xscale=log2,
            yscale=log2,
            yticks=y_ticks,
        ),
    )

    point_methods = [
        method for method in MUL_METHOD_ORDER
        if any(df_long.method .== method)
    ]
    Legend(
        fig.figure[end+1, :],
        [
            [
                LineElement(color=MUL_LINE_COLORS.DB),
                LineElement(color=MUL_LINE_COLORS.BLAS),
            ],
            [
                MarkerElement(color=MUL_POINT_COLORS[method], marker=:circle)
                for method in point_methods
            ],
        ],
        [MUL_LINE_LABELS, point_methods],
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

mul_comp = read_mul_benchmarks()
plot_mul_benchmarks(mul_comp, MUL_M_SMALL; savepath=resultpath("mul_comp_times_small.pdf"))
plot_mul_benchmarks(mul_comp, MUL_M_LARGE; savepath=resultpath("mul_comp_times_large.pdf"))
