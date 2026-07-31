# DBGenzPaper

This repository contains the code and simulations for the paper. It uses [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl) for reproducible project activation and simulation runs.

## Setup

To ensure reproducibility and install all necessary dependencies, run the following command from the root of the repository:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running Simulations

The project uses a central runner script `scripts/run_simulations.jl` to execute simulations and plotting scripts.

### Basic Usage

To run the default set of simulations (configured for quick testing):

```bash
julia --project=. scripts/run_simulations.jl
```

This will run the targets defined in `scripts/parameters/default.toml`.

### Running Specific Targets

You can run specific simulations or plotting scripts by passing their names as arguments:

```bash
julia --project=. scripts/run_simulations.jl simulation_comparison simulation_sparse
```

**Available Targets:**
- `simulation_comparison`
- `simulation_sparse`
- `simulation_FP32`
- `simulation_precision`
- `simulation_cholesky`
- `simulation_mul!`
- `simulation_richtmyer`
- `simulation_timed`
- `plot_cholesky_benchmarks`
- `plot_mul_benchmarks`
- `plot_richtmyer`
- `plot_comparisons`
- `plot_timed_parts`
- `export_tables`
- `example1`
- `example2`

### Running the Full Paper Simulations

To run the full simulations as presented in the paper (with higher repetition counts), use the `paper.toml` configuration file:

```bash
DBGENZPAPER_SIM_CONFIG=scripts/parameters/paper.toml julia --project=. scripts/run_simulations.jl
```

### Running the Example

To run the practical example (`example1.jl` and `example2.jl`):

```bash
julia --project=. scripts/run_simulations.jl example1
julia --project=. scripts/run_simulations.jl example2
```

Or run the plotting script for it:

```bash
julia --project=. scripts/run_simulations.jl plot_example2
```

## Configuration

Simulation parameters are defined in TOML files in `scripts/parameters/`.

- `default.toml`: Contains parameters for quick runs (reps=1), useful for testing the pipeline.
- `paper.toml`: Contains the parameters used for the paper results (high repetition counts).

You can control which configuration file is used by setting the `DBGENZPAPER_SIM_CONFIG` environment variable.

## Unit Tests

The `run_simulations.jl` script automatically runs the unit tests (`test/runtests.jl`) before executing any simulation targets. This serves to verify the correctness of the code and also warms up Julia's JIT compiler.

To run tests manually:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Important Note for Non-Apple Silicon Users

If you are **not** using an Apple Silicon computer with AppleAccelerate, it is recommended to disable `use_AppleBLAS` to avoid oversubscription with BLAS and Julia's threads.

This option is available in the `qmc_pnorm!` function and other related functions in the `src` folder. Ensure `use_AppleBLAS` is set to `false` in your calls or configuration if you are running on standard Linux/Windows machines with OpenBLAS or MKL.

Please note that this version of the code is slightly refactored compared to the code that was used to generate original results - it uses DrWatson.jl so that individual scripts etc. are easier to run for users not very familiar with Julia.
