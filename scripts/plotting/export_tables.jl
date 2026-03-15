include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using LaTeXStrings
using PrettyTables
using Printf

println("Processing Sparse vs Dense...")
sd_path_times = resultpath("sparse_dense_times.csv")
sd_path_vals = resultpath("sparse_dense_vals.csv")

df_sd_times = CSV.read(sd_path_times, DataFrame)
df_sd_p = df_sd_times[df_sd_times.method.=="pnorm", [:n, :n_pts, :matrix, :min]]
rename!(df_sd_p, :min => :pnorm_min)
df_sd_times = leftjoin(df_sd_times, df_sd_p, on=[:n, :n_pts, :matrix])
df_sd_times.ratio = df_sd_times.min ./ df_sd_times.pnorm_min

df_sd_vals = CSV.read(sd_path_vals, DataFrame)
rename!(df_sd_vals, :median => :value, :se_est => :mean_error)

df_sd_times.method = String.(df_sd_times.method)
df_sd_vals.method = String.(df_sd_vals.method)
df_sd_times.matrix = String.(df_sd_times.matrix)
df_sd_vals.matrix = String.(df_sd_vals.matrix)

cols_times = [:n, :n_pts, :matrix, :method, :min, :ratio]
cols_vals = [:n, :n_pts, :matrix, :method, :value, :sd, :mean_error]

df_sd_merged = innerjoin(
    select(df_sd_times, cols_times),
    select(df_sd_vals, cols_vals),
    on=[:n, :n_pts, :matrix, :method],
)

final_cols_sd = [:n, :n_pts, :matrix, :method, :min, :value, :sd, :mean_error, :ratio]
df_sd_final = select(df_sd_merged, final_cols_sd)

replace!(df_sd_final.method, "pnorm" => "DB-FP32", "pnorm_sparse" => "DB-Sparse", "tlr" => "TLR")
df_sd_final.matrix = replace.(df_sd_final.matrix, "fixed" => "\$\\Sigma_2\$", "sparse" => "\$\\Sigma_3\$")
sort!(df_sd_final, [:matrix, :n, :n_pts, :method])
df_sd_final.matrix = LatexCell.(df_sd_final.matrix)

rename!(df_sd_final,
    :n => "n",
    :n_pts => "m",
    :matrix => "Matrix",
    :method => "Method",
    :min => "Time (s)",
    :value => "Value",
    :sd => "SD",
    :mean_error => "Mean Error",
    :ratio => "Ratio",
)

open(resultpath("table_sparse_dense.tex"), "w") do f
    pretty_table(f, df_sd_final;
        backend=:latex,
        formatters=[
            fmt__printf("%d", [1, 2]),
            fmt__printf("%.4f", [5, 6]),
            fmt__printf("%.2e", [7, 8]),
            fmt__printf("%.2f", [9]),
        ],
    )
end

lines = readlines(resultpath("table_sparse_dense.tex"))
filter!(l -> !occursin("\\textit{", l), lines)
open(resultpath("table_sparse_dense.tex"), "w") do f
    for l in lines
        println(f, l)
    end
end

println("Processing Method Comparison...")
cmp_path_times = resultpath("comparisson_times.csv")
cmp_path_vals = resultpath("comparisson_vals.csv")

df_cmp_times = CSV.read(cmp_path_times, DataFrame)
indicator_cols_cmp = [:n, :n_pts, :sim]
df_cmp_p = df_cmp_times[df_cmp_times.method.=="pnorm", vcat(indicator_cols_cmp, [:min])]
rename!(df_cmp_p, :min => :pnorm_min)
df_cmp_times = leftjoin(df_cmp_times, df_cmp_p, on=indicator_cols_cmp)
df_cmp_times.ratio = df_cmp_times.min ./ df_cmp_times.pnorm_min

df_cmp_vals = CSV.read(cmp_path_vals, DataFrame)
rename!(df_cmp_vals, :median => :value, :se_est => :mean_error)

df_cmp_times.method = String.(df_cmp_times.method)
df_cmp_vals.method = String.(df_cmp_vals.method)

cols_times_cmp = [:n, :n_pts, :sim, :method, :min, :ratio]
cols_vals_cmp = [:n, :n_pts, :sim, :method, :value, :sd, :mean_error]

df_cmp_merged = innerjoin(
    select(df_cmp_times, cols_times_cmp),
    select(df_cmp_vals, cols_vals_cmp),
    on=[:n, :n_pts, :sim, :method],
)

final_cols_cmp = [:n, :n_pts, :method, :min, :value, :sd, :mean_error, :ratio]
df_cmp_final = select(df_cmp_merged, final_cols_cmp)

replace!(df_cmp_final.method, "pnorm" => "DB", "mvnormcdf" => "MvNormCDF.jl", "tlr" => "tlrmvnmvt::GenzBretz")
sort!(df_cmp_final, [:n, :n_pts, :method])

rename!(df_cmp_final,
    :n => "n",
    :n_pts => "m",
    :method => "Method",
    :min => "Time (s)",
    :value => "Value",
    :sd => "SD",
    :mean_error => "Mean Error",
    :ratio => "Ratio",
)

open(resultpath("table_comparison.tex"), "w") do f
    pretty_table(f, df_cmp_final;
        backend=:latex,
        formatters=[
            fmt__printf("%d", [1, 2]),
            fmt__printf("%.4f", [4, 5]),
            fmt__printf("%.2e", [6, 7]),
            fmt__printf("%.2f", [8]),
        ],
    )
end

lines = readlines(resultpath("table_comparison.tex"))
filter!(l -> !occursin("\\textit{", l), lines)
open(resultpath("table_comparison.tex"), "w") do f
    for l in lines
        println(f, l)
    end
end

println("Done!")
