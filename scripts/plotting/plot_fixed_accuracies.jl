comparison_df = include(joinpath(@__DIR__, "..", "simulations", "temp.jl"))

using AlgebraOfGraphics
using CairoMakie
using CategoricalArrays
using DataFrames
using LaTeXStrings


const FIXED_ACCURACY_METHODS = [
    "DB",
    "DB-FP32",
    "DB-Sparse",
    "DB-Sparse-FP32",
    "TLR",
]
const FIXED_ACCURACY_MACHINE_ORDER = ["Apple M2 Ultra", "Intel Xeon"]
const FIXED_ACCURACY_BUDGETS = [
    L"m = 2^{11} \times 12",
    L"m = 2^{11} \times 120",
]

# Match the categorical colors in the sparse-times comparison. DB and DB-FP32
# use its first two colors, while TLR keeps its fifth color.
const FIXED_ACCURACY_METHOD_COLORS = [
    "DB" => Makie.wong_colors()[1],
    "DB-FP32" => Makie.wong_colors()[2],
    "DB-Sparse" => Makie.wong_colors()[3],
    "DB-Sparse-FP32" => Makie.wong_colors()[4],
    "TLR" => Makie.wong_colors()[5],
]

# Constant offsets on the log2 scale keep uncertainty whiskers from occupying
# the same x coordinate. The clean figure still uses the exact dimensions.
const FIXED_ACCURACY_LOG2_OFFSETS = Dict(
    "DB" => -0.22,
    "DB-FP32" => -0.11,
    "DB-Sparse" => 0.0,
    "DB-Sparse-FP32" => 0.11,
    "TLR" => 0.22,
)


function prepare_fixed_accuracy_data(results::AbstractDataFrame=comparison_df)
    method_labels = Dict(
        "pnorm" => "DB",
        "pnorm32" => "DB-FP32",
        "pnorm_sparse" => "DB-Sparse",
        "pnorm_sparse32" => "DB-Sparse-FP32",
        "tlr" => "TLR",
    )
    selected_methods = Set(keys(method_labels))
    keep = in.(string.(results.method), Ref(selected_methods))
    df = DataFrame(results[keep, :]; copycols=true)

    required_columns = [
        :machine,
        :method,
        :n,
        :n_pts,
        :absolute_difference,
        :run_sd,
    ]
    missing_columns = setdiff(required_columns, propertynames(df))
    isempty(missing_columns) ||
        error("Fixed-accuracy results are missing columns: $(missing_columns)")
    all(isfinite.(df.absolute_difference) .& (df.absolute_difference .> 0)) ||
        error("Absolute errors must be finite and positive for the logarithmic axis")
    all(isfinite.(df.run_sd) .& (df.run_sd .>= 0)) ||
        error("Run-to-run SD values must be finite and nonnegative")

    df.method_label = [method_labels[string(method)] for method in df.method]
    df.method_label = categorical(
        df.method_label;
        ordered=true,
        levels=FIXED_ACCURACY_METHODS,
    )

    replace!(df.machine, "M2U" => "Apple M2 Ultra", "Intel" => "Intel Xeon")
    df.machine = categorical(
        string.(df.machine);
        ordered=true,
        levels=FIXED_ACCURACY_MACHINE_ORDER,
    )

    budget_labels = Dict(
        24576 => FIXED_ACCURACY_BUDGETS[1],
        245760 => FIXED_ACCURACY_BUDGETS[2],
    )
    unknown_budgets = setdiff(unique(Int.(df.n_pts)), collect(keys(budget_labels)))
    isempty(unknown_budgets) ||
        error("No fixed-accuracy panel label for QMC budgets $(sort(unknown_budgets))")
    df.budget_label = [budget_labels[Int(n_pts)] for n_pts in df.n_pts]
    df.budget_label = categorical(
        df.budget_label;
        ordered=true,
        levels=FIXED_ACCURACY_BUDGETS,
    )

    # On a logarithmic axis, symmetric SD bars would cross zero for these
    # results. Use a one-sided upper whisker whose length is one run-to-run SD.
    df.lower_error = zeros(nrow(df))
    df.upper_error = copy(df.run_sd)
    df.n_dodged = [
        n * exp2(FIXED_ACCURACY_LOG2_OFFSETS[string(method)])
        for (n, method) in zip(df.n, df.method_label)
    ]

    sort!(df, [:machine, :n_pts, :method_label, :n])
    return df
end


function plot_fixed_accuracies(
    output_path::AbstractString=resultpath("fixed_accuracies.pdf"),
    ;
    include_errorbars::Bool=false,
)
    df = prepare_fixed_accuracy_data()
    n_values = sort(unique(Int.(df.n)))
    n_labels = [latexstring("2^{", round(Int, log2(n)), "}") for n in n_values]
    x_column = include_errorbars ? :n_dodged : :n

    shared_mapping = mapping(
        x_column,
        :absolute_difference;
        color=:method_label => "Method",
        group=:method_label,
        col=:machine => "System",
        row=:budget_label => "QMC budget",
    )

    line_layer = AlgebraOfGraphics.data(df) * shared_mapping *
                 (visual(Lines; linewidth=2) + visual(Scatter; markersize=9))
    plot_spec = line_layer
    if include_errorbars
        error_layer = AlgebraOfGraphics.data(df) * mapping(
            x_column,
            :absolute_difference,
            :lower_error,
            :upper_error;
            color=:method_label => "Method",
            col=:machine => "System",
            row=:budget_label => "QMC budget",
        ) * visual(Errorbars; linewidth=1.2, whiskerwidth=6, alpha=0.4)
        # Draw the translucent whiskers first so the method lines and points
        # remain crisp on top of them.
        plot_spec = error_layer + line_layer
    end

    color_scale = scales(Color=(; palette=FIXED_ACCURACY_METHOD_COLORS))
    figure = draw(
        plot_spec,
        color_scale;
        figure=(size=(1000, 700),),
        axis=(
            xlabel=include_errorbars ?
                   "Dimension (n; methods horizontally offset)" :
                   "Dimension (n)",
            ylabel=include_errorbars ?
                   "Absolute error |median - reference| (upper whisker: run-to-run SD)" :
                   "Absolute error |median - reference|",
            xticks=(n_values, n_labels),
            xscale=log2,
            yscale=log10,
        ),
        legend=(position=:bottom, orientation=:horizontal, titleposition=:left),
    )

    save(output_path, figure)
    println("Exported $(output_path)")
    return figure
end


function plot_fixed_accuracies_with_errors(
    output_path::AbstractString=resultpath("fixed_accuracies_errors.pdf"),
)
    return plot_fixed_accuracies(output_path; include_errorbars=true)
end


if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    plot_fixed_accuracies()
    plot_fixed_accuracies_with_errors()
end
