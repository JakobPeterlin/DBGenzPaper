include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie
using LaTeXStrings
using CategoricalArrays

function timed_parts_path(filename::AbstractString)
    path = resultpath(filename)
    isfile(path) && return path
    legacy = joinpath(PROJECT_ROOT, "results", filename)
    isfile(legacy) && return legacy
    return path
end

section_label_map = Dict(
    "Pre-allocation" => "Pre-allocation",
    "Cholesky" => "Cholesky factorization",
    "Generating QMC points" => "QMC point generation",
    "Affine scrambling" => "Affine scrambling",
    "BLAS mul!" => "BLAS multiplication",
    "Copy to/from buffers" => "Buffer copying",
    "Computation of quantiles" => "Quantile computation",
    "Internal multiplication" => "Internal multiplication",
    "Clculation of CDFs" => "CDF calculation",
)

raw_section_order = String[
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
section_order = [section_label_map[s] for s in raw_section_order]

df_apple = CSV.read(timed_parts_path("qmc_parts_timed.csv"), DataFrame)
df_apple.machine .= "Apple M2 Ultra"
df_intel = CSV.read(timed_parts_path("qmc_parts_timedIntel.csv"), DataFrame)
df_intel.machine .= "Intel Xeon"

df_timeit = vcat(df_apple, df_intel)
df_plot = deepcopy(df_timeit)
df_plot = df_plot[df_plot.Section.!="Total", :]
df_plot = df_plot[df_plot.Section.!="DB loop", :]

df_plot.percent = 100 .* df_plot.Percent
df_plot.section = [get(section_label_map, s, s) for s in df_plot.Section]

present_sections = unique(df_plot.section)
extra_sections = sort(collect(setdiff(present_sections, section_order)))
section_levels = vcat([s for s in section_order if s in present_sections], extra_sections)
df_plot.section = categorical(df_plot.section; ordered=true, levels=section_levels)

machine_order = ["Apple M2 Ultra", "Intel Xeon"]
df_plot.machine = categorical(df_plot.machine; ordered=true, levels=machine_order)

df_plot.n_exp = round.(Int, log2.(df_plot.n))
n_pts_values = sort(unique(df_plot.n_pts))
n_pts_order = string.(n_pts_values)
df_plot.n_pts_label = string.(df_plot.n_pts)
df_plot.n_pts_label = categorical(df_plot.n_pts_label; ordered=true, levels=n_pts_order)

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
        col=:machine,
        row=:n_pts_label => "Number of points",
    ) *
    visual(BarPlot)

fig = draw(
    plt;
    figure=(size=(900, 650),),
    axis=(
        xlabel="n",
        ylabel="Runtime (%)",
        xticks=(1:length(exps), xtick_labels),
        xticklabelrotation=0.0,
    ),
    legend=(title="Section", reverse=true),
)

save(resultpath("qmc_parts_stacked.pdf"), fig)
