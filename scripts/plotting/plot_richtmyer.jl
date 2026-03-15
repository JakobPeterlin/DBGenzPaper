include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie
using LaTeXStrings

results_r = CSV.read(resultpath("results_r.csv"), DataFrame)
results_f = CSV.read(resultpath("results_f.csv"), DataFrame)

results_r.Matrix .= "Random"
results_f.Matrix .= "Fixed"

df = vcat(results_r, results_f)
df.n = Int.(round.(df.n))
df.m = Int.(round.(df.m))
df.m_exp = Int.(round.(log2.(df.m)))

m_exps = sort(unique(df.m_exp))
m_labels = [latexstring("2^{", k, "}") for k in m_exps]

df_long = stack(df, [:sd_s, :sd_r]; variable_name=:Method, value_name=:sd)
df_long.Method = replace.(String.(df_long.Method), "sd_s" => "Sobol", "sd_r" => "Richtmyer")

plt = AlgebraOfGraphics.data(df_long) *
      mapping(
    :m_exp, :sd;
    color=:Method,
    group=:Method,
    col=:n,
    row=:Matrix,
) *
      (visual(Lines) + visual(Scatter))

fig = draw(
    plt;
    axis=(
        xlabel="Number of points",
        ylabel="SD (over 1000 repetitions)",
        xticks=(m_exps, m_labels),
    ),
    legend=(position=:bottom, orientation=:horizontal, titleposition=:left),
)

save(resultpath("qmc_points.pdf"), fig)
