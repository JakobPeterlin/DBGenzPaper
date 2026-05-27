include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using Statistics
using AlgebraOfGraphics
using CairoMakie
using LaTeXStrings

function plot_mul_benchmarks(mul_comp::DataFrame; savepath=nothing)
    df = copy(mul_comp)

    baseline = combine(
        groupby(df[df.method.=="mul!", :], [:n, :m, :BLAS]),
        :min => minimum => :mul_min_min,
    )
    df = leftjoin(df, baseline, on=[:n, :m, :BLAS])
    df = df[.!ismissing.(df.mul_min_min), :]

    df.min .= df.min ./ df.mul_min_min
    df.median .= df.median ./ df.mul_min_min

    df_long = stack(df, [:min, :median], [:n, :m, :BLAS, :method];
        variable_name=:stat, value_name=:time
    )
    df_long.stat = [String(s) for s in df_long.stat]
    df_long.method = replace(df_long.method, "pnorm" => "DB Algorithm")

    ns_sorted = sort!(unique(df_long.n))
    log2_ns = Int.(round.(log2.(ns_sorted)))
    tick_labels = [L"2^{%$k}" for k in log2_ns]

    plt = AlgebraOfGraphics.data(df_long) *
          AlgebraOfGraphics.mapping(
        :n, :time;
        color=:method => "Method",
        marker=:stat => "Stat",
        col=:BLAS,
        row=:m,
    ) *
          AlgebraOfGraphics.visual(ScatterLines)

    fig = draw(
        plt;
        figure=(size=(900, 650),),
        axis=(
            xticks=(ns_sorted, tick_labels),
            xlabel="Matrix size (n)",
            ylabel="Standardized time (÷ min mul!)",
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

mul_comp = CSV.read(sim_resultpath("mul_comp.csv"), DataFrame)
plot_mul_benchmarks(mul_comp; savepath=resultpath("mul_comp_times.pdf"))
