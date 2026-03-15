include(joinpath(@__DIR__, "setup.jl"))

using DataFrames, CSV, YFinance, Dates, ProgressMeter, HTTP, JSON, Logging


holdings_snp = CSV.read(datapath("holdings_SnP.csv"), DataFrame; delim=';', header=7)
holdings_aw = CSV.read(datapath("holdings_AW.csv"), DataFrame; delim=';', header=7)


"""
Normalize common ticker formatting issues for Yahoo Finance / yfinance-style APIs.

- trims whitespace
- replaces `/` share-class separator with `-` (e.g. `BRK/B` -> `BRK-B`)
"""
normalize_yahoo_symbol(::Missing) = ""
normalize_yahoo_symbol(x) = replace(strip(string(x)), "/" => "-")


function get_yahoo_symbol_from_name(query_string; allowed_quote_types=("EQUITY", "ETF"))
    # Yahoo's hidden search API
    if ismissing(query_string)
        return nothing
    end
    q = strip(String(query_string))
    if isempty(q)
        return nothing
    end

    url = "https://query2.finance.yahoo.com/v1/finance/search?q=$(HTTP.escapeuri(q))"

    try
        # Make the request with a User-Agent to avoid getting blocked
        response = HTTP.get(url, [
            "User-Agent" => "Mozilla/5.0",
            "Accept" => "application/json",
        ])
        data = JSON.parse(String(response.body))

        # Yahoo returns a list of quotes. Prefer EQUITY/ETF; fall back to first.
        if haskey(data, "quotes") && !isempty(data["quotes"])
            quotes = data["quotes"]
            idx = findfirst(qt -> get(qt, "quoteType", "") in allowed_quote_types, quotes)
            best_match = isnothing(idx) ? quotes[1] : quotes[idx]
            symbol = normalize_yahoo_symbol(get(best_match, "symbol", ""))
            return isempty(symbol) ? nothing : symbol
        else
            println("No match found for: $q")
            return nothing
        end
    catch e
        println("Error searching for $q: $(typeof(e)) - $e")
        return nothing
    end
end





# Non-US tickers need to be renamed
holdings_aw[!, :yticker] = fill("", nrow(holdings_aw))
@showprogress for i in 1:nrow(holdings_aw)

    sym = get_yahoo_symbol_from_name(holdings_aw[i, "Ticker"])
    if isnothing(sym)
        sleep(0.05)
        x = get_yahoo_symbol_from_name(holdings_aw[i, "Holding name"])
        sym = isnothing(x) ? "" : x
        sleep(0.05) # be kind to Yahoo search endpoint
    end
    holdings_aw.yticker[i] = sym
end








CSV.write(datapath("aw.csv"), holdings_aw[holdings_aw.yticker.!="", :])






holdings_aw2 = holdings_aw[holdings_aw.yticker.!="", :]


pr1 = get_prices("AAPL", Date("2025-1-1"), Date("2025-12-31"))


function _timestamps_to_datevec(ts)
    isempty(ts) && return Date[]
    x = ts[1]
    if x isa Date
        return Date.(ts)
    elseif x isa DateTime
        return Date.(ts)
    elseif x isa Integer
        return Date.(unix2datetime.(ts))  # unix seconds -> DateTime -> Date
    elseif x isa AbstractString
        return Date.(DateTime.(ts))
    else
        return Date.(ts)
    end
end

function _get_prices_retry(sym::AbstractString, date1::Date, date2::Date; tries::Int=3)
    last_err = nothing
    for attempt in 1:tries
        try
            return get_prices(sym, date1, date2)
        catch e
            last_err = e
            sleep((0.5 * 2.0^(attempt - 1)) + rand() * 0.25)
        end
    end
    throw(last_err)
end

function get_multiple_prices(tickers, date1=Date("2000-1-1"), date2=Date("2025-12-31"); tries::Int=3)

    diff_length = (date2 - date1).value
    dates = [date1 + Day(i) for i in 0:diff_length]
    df = DataFrame(date=dates)
    fails = 0

    @showprogress for t in tickers
        sleep(0.15)

        try
            sym = normalize_yahoo_symbol(t)
            t_prices = _get_prices_retry(sym, date1, date2; tries=tries)
            dts = _timestamps_to_datevec(t_prices["timestamp"])
            if isempty(dts)
                @warn "Empty price series" ticker = sym
                fails += 1
                continue
            end
            t_df = DataFrame(date=dts)
            t_df[!, Symbol(sym)] = t_prices["close"]
            df = leftjoin(df, t_df, on=:date)
        catch e
            println("Failed to get data for: $(normalize_yahoo_symbol(t)) :: $(typeof(e)) - $e")
            fails += 1
        end
    end

    return df, fails
end



##
# If you want S&P too, uncomment the next 2 lines.
snp_data = get_multiple_prices(holdings_snp.Ticker)
CSV.write(datapath("data_snp.csv"), snp_data[1])

##

aw_data = get_multiple_prices(String.(holdings_aw2.yticker))
CSV.write(datapath("data_aw.csv"), aw_data[1])

