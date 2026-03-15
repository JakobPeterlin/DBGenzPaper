include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie
using JSON
using LaTeXStrings

function add_n_ticks(df::DataFrame; n_col::Symbol)
    n_vals = collect(skipmissing(df[!, n_col]))
    ns_sorted = sort(unique(n_vals))
    log2_ns = Int.(round.(log2.(ns_sorted)))
    tick_labels = [L"2^{%$k}" for k in log2_ns]
    return ns_sorted, tick_labels
end

# Sparse vs fixed matrix comparison (faceted type by matrix)
sd_path = resultpath("sparse_dense_times.csv")
indicator_cols = [:n, :n_pts, :matrix]

df_sd = CSV.read(sd_path, DataFrame)
df_sd_p = df_sd[df_sd.method.=="pnorm", vcat(indicator_cols, [:min])]
rename!(df_sd_p, :min => :pnorm_min)
df_sd = leftjoin(df_sd, df_sd_p, on=indicator_cols)
df_sd.ratio = df_sd.min ./ df_sd.pnorm_min
df_sd.method = string.(df_sd.method)
replace!(df_sd.method, "pnorm" => "DB-FP32", "pnorm_sparse" => "DB-Sparse", "tlr" => "TLR")
df_sd.matrix = latexstring.(df_sd.matrix)
replace!(df_sd.matrix, L"fixed" => L"$\Sigma_2$", L"sparse" => L"$\Sigma_3$")

df_sd.n_pts_str = latexstring.(string.(df_sd.n_pts))
replace!(df_sd.n_pts_str, L"24576" => L"$2^{11} * 12$", L"245760" => L"$2^{11} * 120$",)
n_levels_sd, n_labels_sd = add_n_ticks(df_sd; n_col=:n)

plt_sd = AlgebraOfGraphics.data(df_sd) *
         mapping(
    :n, :ratio;
    color=:method => "Method",
    group=[:method, :n_pts_str],
    col=:matrix => "Matrix",
    row=:n_pts_str => "m",
) *
         (visual(Lines) + visual(Scatter))

fig_sd = draw(
    plt_sd;
    axis=(
        xlabel="Matrix size (n)",
        ylabel="Time ratio (method / DB_FP32)",
        xticks=(n_levels_sd, n_labels_sd),
        xscale=log2,
        yscale=log10,
    ),
    legend=(position=:bottom, orientation=:horizontal, titleposition=:left),
)

save(resultpath("sparse_dense_times_ratio.pdf"), fig_sd)

# Method comparison (no facets)
cmp_path = resultpath("comparisson_times.csv")

indicator_cols = [:n, :n_pts, :sim]
df_cmp = CSV.read(cmp_path, DataFrame)
df_cmp.method = string.(df_cmp.method)
df_cmp_p = df_cmp[df_cmp.method.=="pnorm", vcat(indicator_cols, [:min])]
rename!(df_cmp_p, :min => :pnorm_min)
df_cmp = leftjoin(df_cmp, df_cmp_p, on=indicator_cols)
df_cmp.ratio = df_cmp.min ./ df_cmp.pnorm_min
df_cmp.n_pts_str = string.(df_cmp.n_pts)
n_levels_cmp, n_labels_cmp = add_n_ticks(df_cmp; n_col=:n)
df_cmp.n_pts_str = latexstring.(df_cmp.n_pts_str)
replace!(df_cmp.n_pts_str, L"24576" => L"$2^{11} * 12$", L"245760" => L"$2^{11} * 120$")

df_cmp.method = String.(df_cmp.method)
replace!(df_cmp.method, "pnorm" => "DB", "mvnormcdf" => "MvNormCDF.jl", "tlr" => "tlrmvnmvt::GenzBretz")

plt_cmp = AlgebraOfGraphics.data(df_cmp) *
          mapping(
    :n, :ratio;
    color=:method => "Method",
    group=:method,
    row=:n_pts_str => "m",
) *
          (visual(Lines) + visual(Scatter))

fig_cmp = draw(
    plt_cmp;
    axis=(
        xlabel="Matrix size (n)",
        ylabel="Time ratio (method / DF)",
        xticks=(n_levels_cmp, n_labels_cmp),
        xscale=log2,
        yscale=log10,
    ),
    legend=(position=:bottom, orientation=:horizontal, titleposition=:left),
)

save(resultpath("comparisson_times_ratio.pdf"), fig_cmp)
