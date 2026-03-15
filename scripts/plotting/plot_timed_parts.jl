include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie
using LaTeXStrings
using CategoricalArrays

df_timeit = CSV.read(resultpath("qmc_parts_timed.csv"), DataFrame)
df_plot = deepcopy(df_timeit)
df_plot = df_plot[df_plot.Section.!="Total", :]
df_plot = df_plot[df_plot.Section.!="DB loop", :]

df_plot.percent = 100 .* df_plot.Percent
df_plot.section = df_plot.Section

section_order = String[
    "Pre-allocation",
    "Cholesky",
    "Generating QMC points",
    "Affine scrambling",
    "BLAS mul!",
    "Copy to/from buffers",
    "Computation of quantiles",
    "Internal multiplication",
    "Clculation of CDFs",
]
present_sections = unique(df_plot.section)
extra_sections = sort(collect(setdiff(present_sections, section_order)))
section_levels = vcat([s for s in section_order if s in present_sections], extra_sections)
df_plot.section = categorical(df_plot.section; ordered=true, levels=section_levels)

df_plot.n_exp = round.(Int, log2.(df_plot.n))
df_plot.n_pts_exp = round.(Int, log2.(df_plot.n_pts))
df_plot.n_pts_label = LaTeXString.(["2^{$e}" for e in df_plot.n_pts_exp])

exps = sort(unique(df_plot.n_exp))
xpos = Dict(e => i for (i, e) in enumerate(exps))
df_plot.x = [xpos[e] for e in df_plot.n_exp]
xtick_labels = LaTeXString.(["2^{$e}" for e in exps])

plt =
    AlgebraOfGraphics.data(df_plot) *
    mapping(
        :x => "n",
        :percent => "Runtime (%)";
        stack=:section,
        color=:section,
        col=:n_pts_label => "Number of points",
    ) *
    visual(BarPlot)

fig = draw(
    plt;
    axis=(
        xlabel="n",
        ylabel="Runtime (%)",
        xticks=(1:length(exps), xtick_labels),
        xticklabelrotation=0.0,
    ),
    legend=(title="Section",),
)

save(resultpath("qmc_parts_stacked.pdf"), fig)
