using DrWatson
using TOML
using Pkg
using LinearAlgebra

@quickactivate "DBGenzPaper"
using DBGenzPaper: CPU_post_fix, use_MKL_instead_of_ACC

const PROJECT_ROOT = projectdir()
const DEFAULT_SIM_CONFIG = joinpath(PROJECT_ROOT, "scripts", "parameters", "default.toml")
const SIM_CONFIG_PATH = get(ENV, "DBGENZPAPER_SIM_CONFIG", DEFAULT_SIM_CONFIG)
const SIM_CONFIG = isfile(SIM_CONFIG_PATH) ? TOML.parsefile(SIM_CONFIG_PATH) : Dict{String,Any}()
const APPLE_ACCELERATE_LIB = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
const ACCELERATED_BLAS_TAG = use_MKL_instead_of_ACC ? "MKL" : "ACC"
const ACCELERATED_BLAS_LABEL = use_MKL_instead_of_ACC ? "MKL" : "Accelerate"
const ACCELERATED_BLAS_FULL_LABEL = use_MKL_instead_of_ACC ? "MKL" : "Apple Accelerate"

datapath(parts...) = datadir(parts...)
resultpath(parts...) = datadir("sims", parts...)

function _cpu_result_filename(filename::AbstractString)
    root, ext = splitext(filename)
    if lowercase(ext) == ".csv" && !isempty(CPU_post_fix) && !endswith(root, CPU_post_fix)
        return string(root, CPU_post_fix, ext)
    end
    return filename
end

function sim_resultpath(parts...)
    isempty(parts) && return resultpath()
    path_parts = string.(parts)
    filename = _cpu_result_filename(path_parts[end])
    return resultpath(path_parts[1:end-1]..., filename)
end

function use_accelerated_blas!()
    if use_MKL_instead_of_ACC
        @eval Main begin
            using MKL, LinearAlgebra
            BLAS.set_num_threads(56)
            if isdefined(MKL, :MKL_jll) && !MKL.MKL_jll.is_available()
                error("use_MKL_instead_of_ACC is true, but MKL is not available for this platform.")
            end
            if isdefined(MKL, :lbt_forward_to_mkl)
                MKL.lbt_forward_to_mkl()
            end
        end
    else
        @eval Main using AppleAccelerate
        LinearAlgebra.BLAS.lbt_forward(APPLE_ACCELERATE_LIB; clear=true, suffix_hint="\x1a\$NEWLAPACK")
        LinearAlgebra.BLAS.lbt_forward(APPLE_ACCELERATE_LIB; clear=false, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
    end

    return LinearAlgebra.BLAS.get_config()
end

mkpath(resultpath())

function _to_string_key_dict(x)
    if x isa AbstractDict
        out = Dict{String,Any}()
        for (k, v) in x
            out[string(k)] = _to_string_key_dict(v)
        end
        return out
    elseif x isa AbstractVector
        return [_to_string_key_dict(v) for v in x]
    else
        return x
    end
end

function simcfg(section::AbstractString, key::AbstractString, default)
    section_dict = get(SIM_CONFIG, section, Dict{String,Any}())
    section_dict = _to_string_key_dict(section_dict)
    value = get(section_dict, key, default)
    return value
end
