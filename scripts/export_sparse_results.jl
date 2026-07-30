include(joinpath(@__DIR__, "plotting", "export_tables.jl"))

using AlgebraOfGraphics
using CairoMakie
using CategoricalArrays
using LaTeXStrings

const SPARSE_METHODS = Set(["pnorm_sparse", "pnorm_sparse32"])
const SPARSE_RESULT_KEYS = [:method, :n, :n_pts, :matrix]
const SPARSE_MACHINE_ORDER = ["Apple M2 Ultra", "Intel Xeon"]
const SPARSE_MATRIX_LABELS = Dict(
    "mattern_cov2" => L"$\Sigma_4$",
    "fixed_dense" => L"$\Sigma_2$",
    "mattern_cov1" => L"$\Sigma_3$",
)
const SPARSE_MATRIX_ORDER = [L"$\Sigma_2$", L"$\Sigma_3$", L"$\Sigma_4$"]
const SPARSE_METHOD_ORDER = ["DB", "DB-FP32", "DB-Sparse", "DB-Sparse-FP32", "TLR"]
const SPARSE_RATIO_REF_LINES =
    mapping([50, 100, 250]) * visual(HLines; linestyle=:dash, color=:gray)

function sparse_result_path(filename::AbstractString)
    path = resultpath(filename)
    isfile(path) && return path

    legacy = joinpath(PROJECT_ROOT, "results", filename)
    isfile(legacy) && return legacy

    error("Result file not found: $(path)")
end

function replace_sparse_rows(base_filename::AbstractString, replacement_filename::AbstractString)
    base = CSV.read(sparse_result_path(base_filename), DataFrame)
    replacement = CSV.read(sparse_result_path(replacement_filename), DataFrame)

    names(base) == names(replacement) ||
        error("Schemas differ between $(base_filename) and $(replacement_filename)")

    replacement_methods = Set(string.(replacement.method))
    replacement_methods == SPARSE_METHODS ||
        error("$(replacement_filename) must contain only $(sort!(collect(SPARSE_METHODS)))")

    any(nonunique(replacement, SPARSE_RESULT_KEYS)) &&
        error("$(replacement_filename) contains duplicate sparse-result keys")

    base_sparse = base[in.(string.(base.method), Ref(SPARSE_METHODS)), :]
    base_keys = select(base_sparse, SPARSE_RESULT_KEYS)
    replacement_keys = select(replacement, SPARSE_RESULT_KEYS)
    base_only = antijoin(base_keys, replacement_keys; on=SPARSE_RESULT_KEYS)
    replacement_only = antijoin(replacement_keys, base_keys; on=SPARSE_RESULT_KEYS)
    if !isempty(base_only) || !isempty(replacement_only)
        error(
            "$(replacement_filename) does not exactly cover the sparse rows in " *
            "$(base_filename) (base-only: $(nrow(base_only)), replacement-only: " *
            "$(nrow(replacement_only)))",
        )
    end

    non_sparse = base[.!in.(string.(base.method), Ref(SPARSE_METHODS)), :]
    return vcat(non_sparse, replacement; cols=:setequal)
end

function read_sparse_machine_results(apple::DataFrame, intel::DataFrame)
    apple = copy(apple)
    intel = copy(intel)
    apple.machine .= SPARSE_MACHINE_ORDER[1]
    intel.machine .= SPARSE_MACHINE_ORDER[2]

    df = vcat(apple, intel; cols=:union)
    df.machine = categorical(
        df.machine;
        ordered=true,
        levels=SPARSE_MACHINE_ORDER,
    )
    return df
end

function n_ticks(df::DataFrame)
    ns = sort(unique(collect(skipmissing(df.n))))
    labels = [L"2^{%$k}" for k in Int.(round.(log2.(ns)))]
    return ns, labels
end

function ratio_ticks(values)
    positive = filter(x -> isfinite(x) && x > 0, collect(skipmissing(values)))
    exponents = floor(Int, log10(minimum(positive))):ceil(Int, log10(maximum(positive)))
    values = 10.0 .^ exponents
    labels = [L"10^{%$k}" for k in exponents]

    for (value, label) in [(50.0, L"50"), (250.0, L"250")]
        if !any(isapprox(value; rtol=0), values)
            push!(values, value)
            push!(labels, label)
        end
    end

    order = sortperm(values)
    return values[order], labels[order]
end

function export_sparse_figure(apple_times::DataFrame, intel_times::DataFrame)
    indicator_cols = [:machine, :n, :n_pts, :matrix]
    df = read_sparse_machine_results(apple_times, intel_times)
    df = df[df.n_pts.==2^11 * 120, :]

    baseline = df[df.method.=="pnorm", vcat(indicator_cols, [:min])]
    rename!(baseline, :min => :pnorm_min)
    df = leftjoin(df, baseline; on=indicator_cols)
    df.ratio = df.min ./ df.pnorm_min

    df.method = string.(df.method)
    replace!(
        df.method,
        "pnorm" => "DB",
        "pnorm32" => "DB-FP32",
        "pnorm_sparse" => "DB-Sparse",
        "pnorm_sparse32" => "DB-Sparse-FP32",
        "tlr" => "TLR",
    )
    df = df[in.(df.method, Ref(SPARSE_METHOD_ORDER)), :]
    df.method = categorical(df.method; ordered=true, levels=SPARSE_METHOD_ORDER)
    df.matrix_label = [
        get(SPARSE_MATRIX_LABELS, string(matrix), latexstring(string(matrix)))
        for matrix in df.matrix
    ]
    df.matrix_label = categorical(
        df.matrix_label;
        ordered=true,
        levels=SPARSE_MATRIX_ORDER,
    )

    n_values, n_labels = n_ticks(df)
    y_values, y_labels = ratio_ticks(df.ratio)
    plot = AlgebraOfGraphics.data(df) *
           mapping(
               :n, :ratio;
               color=:method => "Method",
               group=:method,
               col=:machine => "System",
               row=:matrix_label => "Matrix",
           ) *
           (visual(Lines) + visual(Scatter)) +
           SPARSE_RATIO_REF_LINES

    figure = draw(
        plot;
        figure=(size=(900, 750),),
        axis=(
            xlabel="Matrix size (n)",
            ylabel="Time ratio (method / DB)",
            xticks=(n_values, n_labels),
            xscale=log2,
            yscale=log10,
            yticks=(y_values, y_labels),
        ),
        legend=(position=:bottom, orientation=:horizontal, titleposition=:left),
    )

    output = resultpath("sparse_dense_times_ratio.pdf")
    save(output, figure)
    println("Exported $(output)")
end

function main_sparse()
    apple_times = replace_sparse_rows(
        "sparse_dense_times.csv",
        "sparse_dense_times_just.csv",
    )
    intel_times = replace_sparse_rows(
        "sparse_dense_timesIntel.csv",
        "sparse_dense_times_justIntel.csv",
    )
    apple_vals = replace_sparse_rows(
        "sparse_dense_vals.csv",
        "sparse_dense_vals_just.csv",
    )
    intel_vals = replace_sparse_rows(
        "sparse_dense_valsIntel.csv",
        "sparse_dense_vals_justIntel.csv",
    )

    export_sparse_figure(apple_times, intel_times)

    export_sparse_dense_table(apple_times, apple_vals, "table_sparse_dense.tex")
    println("Exported $(resultpath("table_sparse_dense.tex"))")

    export_sparse_dense_table(
        intel_times,
        intel_vals,
        "table_sparse_denseIntel.tex",
    )
    println("Exported $(resultpath("table_sparse_denseIntel.tex"))")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main_sparse()
end
