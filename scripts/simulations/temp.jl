include("setup.jl")

using CSV
using DataFrames
using Distributions


const FIXED_DENSE_SOURCES = (
    (
        option="original",
        stem="sparse_dense_vals",
        methods=("pnorm", "tlr"),
    ),
    (
        option="sparse_rerun",
        stem="sparse_dense_vals_just",
        methods=("pnorm_sparse",),
    ),
    (
        option="fp32_rerun",
        stem="sparse_dense_vals_FP32",
        methods=("pnorm32", "pnorm_sparse32"),
    ),
)

const FIXED_DENSE_MACHINES = (
    (machine="M2U", suffix=""),
    (machine="Intel", suffix="Intel"),
)


"""
    fixed_dense_true_value(n; independent_variance=1, common_variance=1,
                           target_independent_probability=0.25, rtol=1e-12)

Compute the true rectangle probability used by the `fixed_dense` simulations.
For `Σ = αI + β11'`, condition on the common standard-normal factor to reduce
the calculation to a one-dimensional `Distributions.expectation`.
"""
function fixed_dense_true_value(
    n::Integer;
    independent_variance::Real=1.0,
    common_variance::Real=1.0,
    target_independent_probability::Real=0.25,
    rtol::Real=1e-12,
)
    n > 0 || throw(ArgumentError("n must be positive"))
    independent_variance > 0 ||
        throw(ArgumentError("independent_variance must be positive"))
    common_variance >= 0 ||
        throw(ArgumentError("common_variance must be nonnegative"))
    0 < target_independent_probability < 1 ||
        throw(ArgumentError("target_independent_probability must lie in (0, 1)"))

    standard_normal = Normal()
    marginal_variance = independent_variance + common_variance
    k = quantile(
        standard_normal,
        (1 + target_independent_probability^(1 / n)) / 2,
    )
    half_width = sqrt(marginal_variance) * k
    inv_independent_sd = inv(sqrt(float(independent_variance)))
    common_sd = sqrt(float(common_variance))

    return Distributions.expectation(standard_normal; rtol=rtol) do z
        shift = common_sd * z
        standardized_upper = (half_width - shift) * inv_independent_sd
        standardized_lower = (-half_width - shift) * inv_independent_sd
        log_conditional_mass = logdiffcdf(
            standard_normal,
            standardized_upper,
            standardized_lower,
        )
        return exp(n * log_conditional_mass)
    end
end


function fixed_dense_result_path(stem::AbstractString, suffix::AbstractString)
    filename = string(stem, suffix, ".csv")
    path = resultpath(filename)
    isfile(path) || error("Dense/sparse result file not found: $(path)")
    return path
end


function load_fixed_dense_results()
    frames = DataFrame[]

    for machine in FIXED_DENSE_MACHINES
        for source in FIXED_DENSE_SOURCES
            path = fixed_dense_result_path(source.stem, machine.suffix)
            result = CSV.read(path, DataFrame)

            required_columns = [:method, :median, :n, :n_pts, :matrix]
            all(in.(required_columns, Ref(propertynames(result)))) ||
                error("$(basename(path)) does not have the expected value-result schema")

            method_set = Set(source.methods)
            keep = (string.(result.matrix) .== "fixed_dense") .&
                   in.(string.(result.method), Ref(method_set))
            selected = result[keep, :]

            found_methods = Set(string.(selected.method))
            found_methods == method_set || error(
                "$(basename(path)) is missing fixed_dense methods " *
                "$(sort!(collect(setdiff(method_set, found_methods))))",
            )

            selected.machine .= machine.machine
            selected.option .= source.option
            selected.source_file .= basename(path)
            push!(frames, selected)
        end
    end

    results = vcat(frames...; cols=:setequal)
    result_keys = [:machine, :method, :n, :n_pts]
    any(nonunique(results, result_keys)) &&
        error("Selected dense/sparse results contain duplicate settings")

    setting_keys = [:machine, :n, :n_pts]
    baseline = unique(select(results[results.method.=="pnorm", :], setting_keys))
    for method in sort(unique(string.(results.method)))
        settings = unique(select(results[string.(results.method).==method, :], setting_keys))
        isempty(antijoin(baseline, settings; on=setting_keys)) &&
            isempty(antijoin(settings, baseline; on=setting_keys)) ||
            error("Method $(method) does not cover every M2U/Intel fixed_dense option")
    end

    return results
end


function fixed_dense_accuracy_dataframe()
    results = load_fixed_dense_results()

    dimensions = sort(unique(Int.(results.n)))
    truth = DataFrame(
        n=dimensions,
        true_value=fixed_dense_true_value.(dimensions),
    )

    comparison = leftjoin(results, truth; on=:n, validate=(false, true))
    comparison.absolute_difference = abs.(comparison.median .- comparison.true_value)

    # The stored simulations contain the median across repetitions, not the
    # arithmetic mean. Preserve that distinction explicitly in the output.
    rename!(comparison, :median => :median_value, :sd => :run_sd)
    comparison.summary_statistic .= "median"
    comparison.machine_order = ifelse.(comparison.machine .== "M2U", 1, 2)
    comparison.method_order = [
        findfirst(==(method), ["pnorm", "pnorm32", "pnorm_sparse", "pnorm_sparse32", "tlr"])
        for method in string.(comparison.method)
    ]

    sort!(comparison, [:machine_order, :n, :n_pts, :method_order])

    select!(
        comparison,
        :machine,
        :option,
        :source_file,
        :method,
        :n,
        :n_pts,
        :summary_statistic,
        :median_value,
        :run_sd,
        :true_value,
        :absolute_difference,
    )

    return comparison
end


comparison_df = fixed_dense_accuracy_dataframe()

comparison_df
