using DrWatson
using TOML
using Pkg

@quickactivate "DBGenzPaper"

const PROJECT_ROOT = projectdir()
const DEFAULT_SIM_CONFIG = joinpath(PROJECT_ROOT, "scripts", "parameters", "default.toml")
const SIM_CONFIG_PATH = get(ENV, "DBGENZPAPER_SIM_CONFIG", DEFAULT_SIM_CONFIG)
const SIM_CONFIG = isfile(SIM_CONFIG_PATH) ? TOML.parsefile(SIM_CONFIG_PATH) : Dict{String,Any}()

datapath(parts...) = datadir(parts...)
resultpath(parts...) = datadir("sims", parts...)

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
