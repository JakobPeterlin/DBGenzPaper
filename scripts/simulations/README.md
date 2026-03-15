# Running Simulations

All scripts in this folder share `scripts/simulations/setup.jl`, which:

- activates the project with DrWatson (`@quickactivate "DBGenzPaper"`),
- loads parameter configuration from `scripts/parameters/default.toml`,
- defines `datapath(...)` and `resultpath(...)`.

Input files are read from `data/`, and simulation outputs are written to `data/sims/`.

## Reproducible pipeline

From the repository root, run:

```bash
julia --project=. scripts/run_simulations.jl
```

This executes targets from `[runner].targets` in `scripts/parameters/default.toml`.

To run only specific targets:

```bash
julia --project=. scripts/run_simulations.jl simulation_comparison simulation_sparse
```

To use a different parameter file:

```bash
DBGENZPAPER_SIM_CONFIG=path/to/params.toml julia --project=. scripts/run_simulations.jl
```
