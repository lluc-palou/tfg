# ========================================================================================================== #
# Trade Targets Derivation                                                                                   #
# ========================================================================================================== #

def calculate_forward_log_returns(df: DataFrame, lag: int, N: int) -> DataFrame:
    """
    Calculates forward log returns of last traded price over N periods, accounting for a decision lag.
    """
    w = Window.orderBy("timestamp")
    base = lead(col("last_traded_price"), lag).over(w)
    future = lead(col("last_traded_price"), lag + N).over(w)

    return df.withColumn(f"fwd_logret_{N}", log(future) - log(base))

def derive_trade_targets(trades: DataFrame) -> DataFrame:
    """
    Derives trade targets as forward log-returns of last traded price over multiple horizons, accounting for a decision lag.
    """
    H = [2, 3, 4, 5, 10, 20, 40, 60, 120, 240]
    lag = 1

    for N in H:
        trades = calculate_forward_log_returns(trades, lag=lag, N=N)

    return trades