include(joinpath(@__DIR__, "export_tables.jl"))

using Statistics

const PRECISION_INPUTS = [
    ("precision.csv", "M2 Ultra"),
    ("precision_Intel.csv", "Intel Xeon Gold"),
]

const PRECISION_FUNCTION_LABELS = Dict(
    "fixed" => "\$\\Sigma_2\$",
    "fixed_dense" => "\$\\Sigma_2\$",
    "mattern_cov1" => "\$\\Sigma_3\$",
    "matter_cov2" => "\$\\Sigma_4\$",
    "mattern_cov2" => "\$\\Sigma_4\$",
)

const PRECISION_REQUIRED_COLUMNS = [
    :simulation_function,
    :n,
    :absolute_difference,
    :qmc_data_time,
    :qmc_data_distributions_time,
]

const PRECISION_COLUMN_LABELS = [
    [
        "Function",
        "n",
        "Platform",
        "Median time ratio",
        MultiColumn(2, "Absolute difference"),
    ],
    [
        EmptyCells(3),
        "DB-erfc / DB",
        "Maximum",
        "Median",
    ],
]

function precision_function_label(name)
    name = string(name)
    label = get(PRECISION_FUNCTION_LABELS, name, nothing)
    return isnothing(label) ? name : LatexCell(label)
end

function read_precision_results(filename::AbstractString, platform::AbstractString)
    path = resultpath(filename)
    isfile(path) || error("Precision result file not found: $(path)")

    df = CSV.read(path, DataFrame)
    missing_columns = setdiff(PRECISION_REQUIRED_COLUMNS, propertynames(df))
    isempty(missing_columns) ||
        error("$(filename) is missing columns: $(join(string.(missing_columns), ", "))")

    any(ismissing, df.qmc_data_time) &&
        error("$(filename) contains missing DB times")
    any(ismissing, df.qmc_data_distributions_time) &&
        error("$(filename) contains missing DB-erfc times")
    all(>(0), df.qmc_data_time) ||
        error("$(filename) contains non-positive DB times")
    all(>(0), df.qmc_data_distributions_time) ||
        error("$(filename) contains non-positive DB-erfc times")

    df[!, :Platform] = fill(platform, nrow(df))
    df[!, :time_ratio] =
        df.qmc_data_distributions_time ./ df.qmc_data_time
    return df
end

"""
    precision_summary(df)

Summarize every `(simulation_function, n, Platform)` setting. The time ratio is
computed for each paired repetition as `DB-erfc / DB` and then summarized by
its median; it is not the ratio of the two separately computed median times.
"""
function precision_summary(df::AbstractDataFrame)
    return combine(
        groupby(df, [:simulation_function, :n, :Platform]; sort=false),
        :time_ratio => median => :median_time_ratio,
        :absolute_difference => maximum => :maximum_absolute_difference,
        :absolute_difference => median => :median_absolute_difference,
    )
end

function suppress_repeated_precision_labels!(df::DataFrame)
    function_labels = Any["" for _ in 1:nrow(df)]
    n_labels = Any["" for _ in 1:nrow(df)]
    previous_function = nothing
    previous_setting = nothing

    for i in 1:nrow(df)
        simulation_function = df.simulation_function[i]
        setting = (simulation_function, df.n[i])

        if simulation_function != previous_function
            function_labels[i] = precision_function_label(simulation_function)
            previous_function = simulation_function
        end

        if setting != previous_setting
            n_labels[i] = df.n[i]
            previous_setting = setting
        end
    end

    df[!, :simulation_function] = function_labels
    df[!, :n] = n_labels
    return df
end

function write_precision_latex_table(
    output_filename::AbstractString,
    df::AbstractDataFrame,
)
    path = resultpath(output_filename)
    open(path, "w") do io
        pretty_table(io, df;
            backend=:latex,
            column_labels=PRECISION_COLUMN_LABELS,
            style=LatexTableStyle(column_label=["textbf"]),
            formatters=[
                fmt__printf("%.2f", [4]),
                fmt__printf("%.2e", [5, 6]),
            ],
        )
    end
    return path
end

function export_precision_table(
    output_filename::AbstractString="table_precision.tex";
    inputs=PRECISION_INPUTS,
)
    frames = [
        read_precision_results(filename, platform)
        for (filename, platform) in inputs
    ]
    df = vcat(frames...; cols=:setequal)
    summary = precision_summary(df)

    function_order = Dict(
        name => i for (i, name) in
        enumerate(unique(string.(df.simulation_function)))
    )
    platform_order = Dict(
        platform => i for (i, (_, platform)) in enumerate(inputs)
    )
    summary[!, :function_order] = [
        function_order[string(name)] for name in summary.simulation_function
    ]
    summary[!, :platform_order] = [
        platform_order[string(platform)] for platform in summary.Platform
    ]
    sort!(summary, [:function_order, :n, :platform_order])
    select!(summary, Not([:function_order, :platform_order]))
    suppress_repeated_precision_labels!(summary)

    rename!(
        summary,
        :simulation_function => "Function",
        :n => "n",
        :median_time_ratio => "Median time ratio (DB-erfc / DB)",
        :maximum_absolute_difference => "Max abs. difference",
        :median_absolute_difference => "Median abs. difference",
    )

    output_path = write_precision_latex_table(output_filename, summary)
    println("Exported $(output_path)")
    return summary
end

function main_precision()
    export_precision_table()
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main_precision()
end
