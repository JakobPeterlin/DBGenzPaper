include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using LaTeXStrings
using PrettyTables
using Printf

const SPARSE_DENSE_METHOD_LABELS = Dict(
    "pnorm" => "DB-FP64",
    "pnorm32" => "DB-FP32",
    "pnorm_sparse" => "DB-Sparse",
    "pnorm_sparse32" => "DB-Sparse-FP32",
    "tlr" => "TLR",
)

const COMPARISON_METHOD_LABELS = Dict(
    "pnorm" => "DB",
    "mvnmvt" => "mvtnorm",
    "mvnormcdf" => "MvNormCDF.jl",
    "tlr" => "tlrmvnmvt::GenzBretz",
)

const MATRIX_LABELS = Dict(
    "fixed" => "\$\\Sigma_2\$",
    "fixed_dense" => "\$\\Sigma_2\$",
    "mattern_cov1" => "\$\\Sigma_3\$",
    "mattern_cov2" => "\$\\Sigma_4\$",
)

const SPARSE_DENSE_MATRIX_ORDER = Dict(
    "mattern_cov2" => 3,
    "fixed" => 1,
    "fixed_dense" => 1,
    "mattern_cov1" => 2,
)

label_methods!(methods, labels) = replace!(methods, [k => v for (k, v) in labels]...)
label_matrix(matrix) = get(MATRIX_LABELS, matrix, matrix)
matrix_order(matrix) = get(SPARSE_DENSE_MATRIX_ORDER, matrix, typemax(Int))

function write_latex_table(filename::AbstractString, df::DataFrame; formatters)
    path = resultpath(filename)
    open(path, "w") do f
        pretty_table(f, df;
            backend=:latex,
            formatters=formatters,
        )
    end

    lines = readlines(path)
    filter!(l -> !occursin("\\textit{", l), lines)
    open(path, "w") do f
        for l in lines
            println(f, l)
        end
    end
end

function export_sparse_dense_table(df_times::AbstractDataFrame, df_vals::AbstractDataFrame,
    output_filename::AbstractString)
    df_times = DataFrame(df_times; copycols=true)
    df_times.method = String.(df_times.method)
    df_times.matrix = String.(df_times.matrix)
    df_p = df_times[df_times.method.=="pnorm", [:n, :n_pts, :matrix, :min]]
    rename!(df_p, :min => :pnorm_min)
    df_times = leftjoin(df_times, df_p, on=[:n, :n_pts, :matrix])
    df_times.ratio = df_times.min ./ df_times.pnorm_min

    df_vals = DataFrame(df_vals; copycols=true)
    rename!(df_vals, :median => :value, :se_est => :mean_error)
    df_vals.method = String.(df_vals.method)
    df_vals.matrix = String.(df_vals.matrix)

    cols_times = [:n, :n_pts, :matrix, :method, :min, :ratio]
    cols_vals = [:n, :n_pts, :matrix, :method, :value, :sd, :mean_error]
    df_merged = innerjoin(
        select(df_times, cols_times),
        select(df_vals, cols_vals),
        on=[:n, :n_pts, :matrix, :method],
    )

    df_final = select(df_merged, [:n, :n_pts, :matrix, :method, :min, :value, :sd, :mean_error, :ratio])
    df_final.matrix_order = matrix_order.(df_final.matrix)
    label_methods!(df_final.method, SPARSE_DENSE_METHOD_LABELS)
    sort!(df_final, [:matrix_order, :matrix, :n, :n_pts, :method])
    select!(df_final, Not(:matrix_order))
    df_final.matrix = LatexCell.(label_matrix.(df_final.matrix))

    rename!(df_final,
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

    write_latex_table(output_filename, df_final;
        formatters=[
            fmt__printf("%d", [1, 2]),
            fmt__printf("%.4f", [5, 6]),
            fmt__printf("%.2e", [7, 8]),
            fmt__printf("%.2f", [9]),
        ],
    )
end

function export_sparse_dense_table(times_filename::AbstractString, vals_filename::AbstractString,
    output_filename::AbstractString)
    df_times = CSV.read(resultpath(times_filename), DataFrame)
    df_vals = CSV.read(resultpath(vals_filename), DataFrame)
    export_sparse_dense_table(df_times, df_vals, output_filename)
end

function export_comparison_table(times_filename::AbstractString, vals_filename::AbstractString,
    output_filename::AbstractString)
    df_times = CSV.read(resultpath(times_filename), DataFrame)
    df_times.method = String.(df_times.method)
    indicator_cols = [:n, :n_pts, :sim]
    df_p = df_times[df_times.method.=="pnorm", vcat(indicator_cols, [:min])]
    rename!(df_p, :min => :pnorm_min)
    df_times = leftjoin(df_times, df_p, on=indicator_cols)
    df_times.ratio = df_times.min ./ df_times.pnorm_min

    df_vals = CSV.read(resultpath(vals_filename), DataFrame)
    rename!(df_vals, :median => :value, :se_est => :mean_error)
    df_vals.method = String.(df_vals.method)

    cols_times = [:n, :n_pts, :sim, :method, :min, :ratio]
    cols_vals = [:n, :n_pts, :sim, :method, :value, :sd, :mean_error]
    df_merged = innerjoin(
        select(df_times, cols_times),
        select(df_vals, cols_vals),
        on=[:n, :n_pts, :sim, :method],
    )

    df_final = select(df_merged, [:n, :n_pts, :method, :min, :value, :sd, :mean_error, :ratio])
    label_methods!(df_final.method, COMPARISON_METHOD_LABELS)
    sort!(df_final, [:n, :n_pts, :method])

    rename!(df_final,
        :n => "n",
        :n_pts => "m",
        :method => "Method",
        :min => "Time (s)",
        :value => "Value",
        :sd => "SD",
        :mean_error => "Mean Error",
        :ratio => "Ratio",
    )

    write_latex_table(output_filename, df_final;
        formatters=[
            fmt__printf("%d", [1, 2]),
            fmt__printf("%.4f", [4, 5]),
            fmt__printf("%.2e", [6, 7]),
            fmt__printf("%.2f", [8]),
        ],
    )
end

const ERF_PLATFORM_ORDER = Dict(
    "M2 Ultra" => 1,
    "Intel Xeon Gold" => 2,
)

const ERF_PRECISION_ORDER = Dict(
    "Float64" => 1,
    "Float32" => 2,
)

const ERF_PRECISION_LABELS = Dict(
    "Float64" => "FP64",
    "Float32" => "FP32",
)

const ERF_PRECISION_EPS_LABELS = Dict(
    "Float64" => "64",
    "Float32" => "32",
)

function format_eps_error(value::Real, precision::AbstractString)
    value_string = if iszero(value)
        "0.000"
    elseif abs(value) < 1e-3 || abs(value) >= 1e3
        @sprintf("%.2e", value)
    else
        @sprintf("%.3f", value)
    end

    return LatexCell("\$" * value_string * " \\epsilon_{" * ERF_PRECISION_EPS_LABELS[precision] * "}\$")
end

function export_erf_table(output_filename::AbstractString)
    df_m2 = CSV.read(resultpath("erf_acc.csv"), DataFrame)
    df_intel = CSV.read(resultpath("erf_accIntel.csv"), DataFrame)

    select!(df_m2, :Precision, :func, :impl, :max_max_error, :mean_max_error, :ratio_mean)
    select!(df_intel, :Precision, :func, :impl, :max_max_error, :mean_max_error, :ratio_mean)
    df_m2.Precision = String.(df_m2.Precision)
    df_intel.Precision = String.(df_intel.Precision)
    df_m2[!, :Platform] = fill("M2 Ultra", nrow(df_m2))
    df_intel[!, :Platform] = fill("Intel Xeon Gold", nrow(df_intel))

    df_erf = vcat(df_m2, df_intel)
    df_erf[!, :precision_order] = [ERF_PRECISION_ORDER[precision] for precision in df_erf.Precision]
    df_erf[!, :platform_order] = [ERF_PLATFORM_ORDER[platform] for platform in df_erf.Platform]
    sort!(df_erf, [:precision_order, :func, :impl, :platform_order])

    precision_labels = fill("", nrow(df_erf))
    function_labels = fill("", nrow(df_erf))
    prev_precision = nothing
    prev_key = nothing
    for i in 1:nrow(df_erf)
        if df_erf.Precision[i] != prev_precision
            precision_labels[i] = ERF_PRECISION_LABELS[df_erf.Precision[i]]
            prev_precision = df_erf.Precision[i]
        end

        key = (df_erf.Precision[i], df_erf.func[i], df_erf.impl[i])
        if key != prev_key
            function_labels[i] = df_erf.impl[i]
            prev_key = key
        end
    end
    select!(df_erf, Not([:precision_order, :platform_order]))

    df_erf = DataFrame(
        "Precision" => precision_labels,
        "Function" => function_labels,
        "Platform" => df_erf.Platform,
        "Max max-abs error" => format_eps_error.(df_erf.max_max_error, df_erf.Precision),
        "Mean max-abs error" => format_eps_error.(df_erf.mean_max_error, df_erf.Precision),
        "Mean time ratio" => df_erf.ratio_mean,
    )

    write_latex_table(output_filename, df_erf;
        formatters=[
            fmt__printf("%.2f", [6]),
        ],
    )
end

function main()
    println("Processing Sparse vs Dense...")
    export_sparse_dense_table("sparse_dense_times.csv", "sparse_dense_vals.csv", "table_sparse_dense.tex")

    println("Processing Sparse vs Dense (Intel)...")
    export_sparse_dense_table("sparse_dense_timesIntel.csv", "sparse_dense_valsIntel.csv", "table_sparse_denseIntel.tex")

    println("Processing Method Comparison...")
    export_comparison_table("comparisson_times.csv", "comparisson_vals.csv", "table_comparison.tex")

    println("Processing Method Comparison (Intel)...")
    export_comparison_table("comparisson_timesIntel.csv", "comparisson_valsIntel.csv", "table_comparisonIntel.tex")

    println("Processing erf/erfinv Comparison...")
    export_erf_table("table_erf_acc.tex")

    println("Done!")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
