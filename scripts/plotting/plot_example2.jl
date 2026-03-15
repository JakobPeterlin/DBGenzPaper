include(joinpath(@__DIR__, "..", "simulations", "setup.jl"))

using CSV
using DataFrames
using AlgebraOfGraphics
using CairoMakie

ps_df = CSV.read(datapath("ps_aw.csv"), DataFrame)

ps_long = vcat(
    DataFrame(b=ps_df.b, market="SNP", val=ps_df.val_snp, err=ps_df.err_snp),
    DataFrame(b=ps_df.b, market="AW", val=ps_df.val_aw, err=ps_df.err_aw),
)
ps_long.low = ps_long.val .- ps_long.err
ps_long.high = ps_long.val .+ ps_long.err

plt_band = AlgebraOfGraphics.data(ps_long) * mapping(:b, :low, :high, color=:market) * visual(Band, alpha=0.4)
plt_line = AlgebraOfGraphics.data(ps_long) * mapping(:b, :val, color=:market) * visual(Lines)
plt_pts = AlgebraOfGraphics.data(ps_long) * mapping(:b, :val, color=:market) * visual(Scatter)

fig = draw(plt_band + plt_line + plt_pts; axis=(xlabel="b", ylabel="p(b)", yscale=log10))
save(resultpath("ps_df_aog.pdf"), fig)
