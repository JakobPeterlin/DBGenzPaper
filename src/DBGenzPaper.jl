module DBGenzPaper

# For running on Intel processors.
const use_MKL_instead_of_ACC = false

# For storing the results on different processors in different files.
const CPU_post_fix = ""

include("qmc.jl")

module SparseQMC
include("qmc_sparse.jl")
end

export CholeskyGenz
export adapt_low_rank!, cholesky_classic!, cholesky_rowmax!, cholesky_genz!
export QMCOpts, QMC_opts
export QMCData, QMCDataLowRank, QMCData_TB, QMCData_TBLowRank, QMCDataSparse
export qmc_pnorm!
export SparseQMC
export use_MKL_instead_of_ACC, CPU_post_fix

end
