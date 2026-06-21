include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie
using JSON
using LaTeXStrings
using CategoricalArrays

const machine_order = ["Apple M2 Ultra", "Intel Xeon"]

function comparison_path(filename::AbstractString)
    path = resultpath(filename)
    isfile(path) && return path
    legacy = joinpath(PROJECT_ROOT, "results", filename)
    isfile(legacy) && return legacy
    return path
end

function read_machine_results(apple_filename::AbstractString, intel_filename::AbstractString)
    df_apple = CSV.read(comparison_path(apple_filename), DataFrame)
    df_apple.machine .= machine_order[1]
    df_intel = CSV.read(comparison_path(intel_filename), DataFrame)
    df_intel.machine .= machine_order[2]

    df = vcat(df_apple, df_intel; cols=:union)
    df.machine = categorical(df.machine; ordered=true, levels=machine_order)
    return df
end

function add_n_ticks(df::DataFrame; n_col::Symbol)
    n_vals = collect(skipmissing(df[!, n_col]))
    ns_sorted = sort(unique(n_vals))
    log2_ns = Int.(round.(log2.(ns_sorted)))
    tick_labels = [L"2^{%$k}" for k in log2_ns]
    return ns_sorted, tick_labels
end

function power2_ticks(values)
    positive_values = filter(x -> isfinite(x) && x > 0, collect(skipmissing(values)))
    exps = floor(Int, log2(minimum(positive_values))):ceil(Int, log2(maximum(positive_values)))
    tick_values = 2.0 .^ exps
    tick_labels = [L"2^{%$k}" for k in exps]
    return tick_values, tick_labels
end

# Sparse vs fixed matrix comparison
indicator_cols = [:machine, :n, :n_pts, :matrix]
sd_max_pts = 2^11 * 120
matrix_label_map = Dict(
    "mattern2" => L"$\Sigma_1$",
    "mattern_cov2" => L"$\Sigma_1$",
    "fixed" => L"$\Sigma_2$",
    "fixed_dense" => L"$\Sigma_2$",
    "mattern1" => L"$\Sigma_3$",
    "mattern_cov1" => L"$\Sigma_3$",
    "sparse" => L"$\Sigma_3$",
)
matrix_order = [L"$\Sigma_1$", L"$\Sigma_2$", L"$\Sigma_3$"]

df_sd = read_machine_results("sparse_dense_times.csv", "sparse_dense_timesIntel.csv")
df_sd = df_sd[df_sd.n_pts.==sd_max_pts, :]
df_sd_p = df_sd[df_sd.method.=="pnorm", vcat(indicator_cols, [:min])]
rename!(df_sd_p, :min => :pnorm_min)
df_sd = leftjoin(df_sd, df_sd_p, on=indicator_cols)
df_sd.ratio = df_sd.min ./ df_sd.pnorm_min
df_sd.method = string.(df_sd.method)
replace!(df_sd.method, "pnorm" => "DB-FP32", "pnorm_sparse" => "DB-Sparse", "tlr" => "TLR")
df_sd = df_sd[in.(df_sd.method, Ref(["DB-FP32", "DB-Sparse", "TLR"])), :]
df_sd.matrix_label = [get(matrix_label_map, string(m), latexstring(string(m))) for m in df_sd.matrix]
df_sd.matrix_label = categorical(df_sd.matrix_label; ordered=true, levels=matrix_order)
n_levels_sd, n_labels_sd = add_n_ticks(df_sd; n_col=:n)
y_ticks_sd = power2_ticks(df_sd.ratio)

plt_sd = AlgebraOfGraphics.data(df_sd) *
         mapping(
    :n, :ratio;
    color=:method => "Method",
    group=:method,
    col=:machine => "System",
    row=:matrix_label => "Matrix",
) *
         (visual(Lines) + visual(Scatter))

fig_sd = draw(
    plt_sd;
    figure=(size=(900, 750),),
    axis=(
        xlabel="Matrix size (n)",
        ylabel="Time ratio (method / DB-FP32)",
        xticks=(n_levels_sd, n_labels_sd),
        xscale=log2,
        yscale=log2,
        yticks=y_ticks_sd,
    ),
    legend=(position=:bottom, orientation=:horizontal, titleposition=:left),
)

save(resultpath("sparse_dense_times_ratio.pdf"), fig_sd)

# Method comparison
indicator_cols = [:machine, :n, :n_pts, :sim]
df_cmp = read_machine_results("comparisson_times.csv", "comparisson_timesIntel.csv")
df_cmp.method = string.(df_cmp.method)
df_cmp_p = df_cmp[df_cmp.method.=="pnorm", vcat(indicator_cols, [:min])]
rename!(df_cmp_p, :min => :pnorm_min)
df_cmp = leftjoin(df_cmp, df_cmp_p, on=indicator_cols)
df_cmp.ratio = df_cmp.min ./ df_cmp.pnorm_min
df_cmp.n_pts_str = string.(df_cmp.n_pts)
n_levels_cmp, n_labels_cmp = add_n_ticks(df_cmp; n_col=:n)
y_ticks_cmp = power2_ticks(df_cmp.ratio)
df_cmp.n_pts_str = latexstring.(df_cmp.n_pts_str)
replace!(df_cmp.n_pts_str, L"24576" => L"$2^{11} * 12$", L"245760" => L"$2^{11} * 120$")

df_cmp.method = String.(df_cmp.method)
replace!(df_cmp.method, "pnorm" => "DB", "mvnormcdf" => "MvNormCDF.jl", "tlr" => "tlrmvtnorm::GenzBretz")

plt_cmp = AlgebraOfGraphics.data(df_cmp) *
          mapping(
    :n, :ratio;
    color=:method => "Method",
    group=:method,
    col=:machine => "System",
    row=:n_pts_str => "m",
) *
          (visual(Lines) + visual(Scatter))

fig_cmp = draw(
    plt_cmp;
    figure=(size=(900, 600),),
    axis=(
        xlabel="Matrix size (n)",
        ylabel="Time ratio (method / DF)",
        xticks=(n_levels_cmp, n_labels_cmp),
        xscale=log2,
        yscale=log2,
        yticks=y_ticks_cmp,
    ),
    legend=(position=:bottom, orientation=:horizontal, titleposition=:left),
)

save(resultpath("comparisson_times_ratio.pdf"), fig_cmp)
