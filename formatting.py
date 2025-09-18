import numpy as np
import pandas as pd
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType, TimestampType
from pyspark.sql.functions import col, expr, from_json, regexp_replace, log, lag, pandas_udf, lit, PandasUDFType, concat, when

# Initializes Spark session with MongoDB connector
jar_files_path = "file:///C:/Users/llucp/spark_jars/"

jar_files = [
    "mongo-spark-connector_2.12-10.1.1.jar",
    "mongodb-driver-core-4.10.1.jar",
    "mongodb-driver-sync-4.10.1.jar",
    "bson-4.10.1.jar"
]

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tfg"
LANDING_TRADE = "landing_trade"
LANDING_LOB = "landing_lob"
FORMATTED_TRADE = "formatting_trade"
FORMATTED_LOB = "formatting_lob"

spark = (
    SparkSession.builder
    .appName("FormattedZone")
    .config("spark.jars", ",".join([jar_files_path + jar for jar in jar_files]))
    .config("spark.mongodb.read.connection.uri", MONGO_URI)
    .config("spark.mongodb.write.connection.uri", MONGO_URI)
    .config("spark.mongodb.read.database", DB_NAME)
    .config("spark.mongodb.write.database", DB_NAME)
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ========================================================================================================== #
# General Data Management                                                                                    #
# ========================================================================================================== #

def load_lob() -> DataFrame:
    """
    Loads Landing Zone LOB data and returns it as a dataframe.
    """
    lob = (
        spark.read.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", LANDING_LOB)
        .load()
    )
    
    # Limits to the most recent 50 LOB snapshots for testing purposes
    lob = lob.orderBy("timestamp").limit(1000)

    return lob

def save_lob(lob: DataFrame) -> None:
    """
    Saves pre-processed LOB dataframe into Formatted Zone (overwrite).
    """
    lob.write.format("mongodb") \
        .option("database", DB_NAME) \
        .option("collection", FORMATTED_LOB) \
        .mode("overwrite") \
        .save()

def load_trade() -> DataFrame:
    """
    Loads Landing Zone TRADE data and returns it as a dataframe.
    """
    trades = (
        spark.read.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", LANDING_TRADE)
        .load()
    )

    # Limits to the most recent 50 trades for testing purposes
    trades = trades.orderBy("timestamp").limit(5000)

    return trades

def save_trade(trades: DataFrame) -> None:
    """
    Saves pre-processed TRADE dataframe into Formatted Zone (overwrite).
    """
    trades.write.format("mongodb") \
        .option("database", DB_NAME) \
        .option("collection", FORMATTED_TRADE) \
        .mode("overwrite") \
        .save()


# ========================================================================================================== #
# LOB Standarization                                                                                         #
# ========================================================================================================== #

def calculate_mid_prices(lob: DataFrame) -> DataFrame:
    """
    Calculates mid-prices from best bid and best ask prices.
    """
    best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
    best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")

    return lob.withColumn("mid_price", (best_bid + best_ask) / 2.0)

def calculate_log_returns(lob: DataFrame) -> DataFrame:
    """
    Calculates log-returns of mid-prices.
    """
    w = Window.orderBy("timestamp")

    return lob.withColumn("log_return", log(col("mid_price")) - log(lag(col("mid_price"), 1).over(w))).dropna()

def estimate_variance(lob: DataFrame, half_life: int) -> DataFrame:
    """
    Estimates mid-price log-returns variance through computing an EWMA of squared mid-price log-returns.
    """
    schema = StructType(
        [
            StructField("timestamp", TimestampType(), True),
            StructField("variance_proxy", DoubleType(), True),
        ]
    ) 

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def calculate_ewma(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("timestamp")
        ts = pd.Series(df["log_return"], dtype="float64")
        df["variance_proxy"] = ts.pow(2).ewm(halflife=half_life, adjust=False).mean()

        return df[["timestamp", "variance_proxy"]]
    
    return (
        lob
        .withColumn("_g", lit(1))
        .groupBy("_g")
        .apply(calculate_ewma)
        .drop("_g")
        .join(lob, on="timestamp", how="right")
    ).dropna()

def standarize_prices(lob: DataFrame) -> DataFrame:
    """
    Standardizes bid and ask prices by subtracting mid-prices and dividing by volatility times the mid-prices.
    This final scaling is due to instead of representing price in log-volatility units, the intend is to represent
    it in volatility-aware relative price units, having as reference the snapshot mid-price. 
    """
    eps = 1e-12
    
    lob = lob \
        .withColumn("bids", expr(f"transform(bids, x -> array((x[0] - mid_price) / ((sqrt(variance_proxy) * mid_price) + {eps}), x[1]))")) \
        .withColumn("asks", expr(f"transform(asks, x -> array((x[0] - mid_price) / ((sqrt(variance_proxy) * mid_price) + {eps}), x[1]))"))
    
    return lob

def normalize_volumes(lob: DataFrame) -> DataFrame:
    """
    Normalizes trading volume, taking into account all snapshot liquidity.
    """
    lob = lob.withColumn(
        "total_volume",
        expr("aggregate(bids, 0D, (acc, x) -> acc + x[1]) + aggregate(asks, 0D, (acc, x) -> acc + x[1])")
    )

    lob = lob \
        .withColumn("bids", expr("transform(bids, x -> array(x[0], IF(total_volume=0, 0D, x[1] / total_volume)))")) \
        .withColumn("asks", expr("transform(asks, x -> array(x[0], IF(total_volume=0, 0D, x[1] / total_volume)))"))

    return lob

def flatten_bids_and_asks(lob: DataFrame) -> DataFrame:
    """
    Flattens bid and ask standarized and normalized volume pairs.
    """
    lob = lob.withColumn("pairs", expr("concat(bids, asks)"))
    lob = lob.withColumn("prices", expr("transform(pairs, x -> x[0])"))
    lob = lob.withColumn("volumes", expr("transform(pairs, x -> x[1])"))

    return lob

def calculate_gamma_scaling_factor(lob: DataFrame, L: float, target: float, gammas_grid: np.ndarray) -> DataFrame:
    """
    Calculates per-snapshot gamma scaling factor through volume coverage optimization. This factor captures liquidity regime.
    """
    @pandas_udf("double", PandasUDFType.SCALAR)
    def calculate_gammas(std_prices: pd.Series, norm_volumes: pd.Series) -> pd.Series:
        gammas = []

        for std_price, norm_volume in zip(std_prices, norm_volumes):
            std_price = np.asarray(std_price, dtype=float)
            norm_volume = np.asarray(norm_volume, dtype=float)
            best_gamma, best_diff = None, float("inf")

            for gamma in gammas_grid:
                cov = norm_volume[np.abs(std_price / gamma) <= L].sum()
                diff = abs(cov - target)

                if diff < best_diff:
                    best_diff, best_gamma = diff, gamma

            gammas.append(best_gamma)

        return pd.Series(gammas, dtype=float)

    return lob.withColumn("gamma", calculate_gammas(col("prices"), col("volumes")))

def gamma_scale_std_prices(lob: DataFrame) -> DataFrame:
    """
    Applies gammas scaling to standarized prices.
    """
    return lob.withColumn("prices", expr("transform(prices, x -> x / gamma)"))

def standarize_lob(lob: DataFrame) -> DataFrame:
    """
    Pipeline: Calculates mid-prices, log-returns, variance proxy, standarizes prices, normalizes volumes, merges bid 
    and ask standarized prices and normalized volumes, calculates gamma scaling factor and scales standarized prices.
    """
    lob = calculate_mid_prices(lob)
    lob = calculate_log_returns(lob)
    lob = estimate_variance(lob, half_life=20)
    lob = standarize_prices(lob)
    lob = normalize_volumes(lob)
    lob = flatten_bids_and_asks(lob)
    lob = calculate_gamma_scaling_factor(lob, L=10, target=0.95, gammas_grid=np.linspace(100, 5000, 80))
    lob = gamma_scale_std_prices(lob)

    return lob

# ========================================================================================================== #
# LOB Binnarization                                                                                          #
# ========================================================================================================== #

def quantize_lob(lob: DataFrame, L: float, B: int, w_factor: float, delta_grid: np.ndarray, clip_max_percentile: float):
    """
    Quantizes the dynamic range of standarized relative prices through optimizing delta hyperparameter, globally, by maximizing 
    volume weighted SNR.
    """
    # Retrieves and flattens standarized prices, normalized volumes and gamma scaling factors
    all_prices, all_volumes, gammas = [], [], []

    for row in lob.select("prices", "volumes", "gamma").collect():
        prices = np.asarray(row["prices"], dtype=float)
        volumes = np.asarray(row["volumes"], dtype=float)
        gamma = row["gamma"]
        all_prices.append(prices)
        all_volumes.append(volumes)
        gammas.append(gamma)

    all_prices = np.concatenate(all_prices).astype(np.float64, copy=False)
    all_volumes = np.concatenate(all_volumes).astype(np.float64, copy=False)

    # Drops NaN values
    mask =  np.isfinite(all_prices) & np.isfinite(all_volumes) & (all_volumes >= 0.0)
    all_prices = all_prices[mask]
    all_volumes = all_volumes[mask]
    
    # Defines a global reference gamma scaling factor to set center width (a fraction of non-scaled variance)
    gamma_ref = np.median(gammas)
    w0 = w_factor / gamma_ref

    # Normalizes volumes globally
    all_volumes = all_volumes / all_volumes.sum()

    # Calculates standarized relative prices threshold to avoid extreme values
    abs_dist_all_prices = np.abs(all_prices)

    def weighted_quantile(x: np.ndarray, w: np.ndarray, p: float) -> float:
        order = np.argsort(x)
        xx, ww = x[order], w[order]
        cdf = np.cumsum(ww)
        idx = np.searchsorted(cdf, p, side="left")

        return float(xx[min(idx, len(xx) - 1)])
    
    prices_max = weighted_quantile(abs_dist_all_prices, all_volumes, clip_max_percentile)

    # Calculates bin edges from quantization process
    bins_per_side = (B - 4) // 2

    def calculate_edges(delta: float):
        y_max = np.log1p((L - w0) / delta)
        y_edges = np.linspace(0.0, y_max, bins_per_side + 1, dtype=np.float64)
        x_pos_edges = w0 + delta * np.expm1(y_edges).astype(np.float64)
        x_neg_edges = (-x_pos_edges[::-1]).astype(np.float64)

        edges = np.concatenate(
            [
                np.array([-prices_max], dtype=np.float64),  # Negative tail edge
                x_neg_edges[:-1],                           # Inner negative ring edges [-L, -w0)
                np.array([-w0, 0.0, w0], dtype=np.float64), # Central edges
                x_pos_edges[1:],                            # Inner positive ring edges (w0, +L])
                np.array([prices_max], dtype=np.float64)    # Positive tail edge
            ]
        ).astype(np.float64, copy=False)

        return np.asarray(edges, dtype=np.float64)
    
    # Optimizes delta to maximize SNR
    eps = 1e-18

    def calculate_snr(delta:float):
        edges = calculate_edges(delta)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        idx = np.digitize(all_prices, edges, right=False) - 1
        idx = np.clip(idx, 0, B-1)
        quantized_std_prices = bin_centers[idx]
        signal = float(np.sum(all_volumes * (all_prices ** 2)))
        mse = float(np.sum(all_volumes * ((all_prices - quantized_std_prices) ** 2)))
        snr = signal / max(mse, eps)
        
        return snr, edges

    best_snr, best_delta, best_edges = -np.inf, None, None

    for delta in delta_grid:
        snr, edges = calculate_snr(delta)
        
        if snr > best_snr:
            best_snr, best_delta, best_edges = snr, delta, edges

    return best_delta, best_edges, best_snr

def aggregate_volume_per_bin(lob: DataFrame, edges: np.ndarray) -> DataFrame:
    """
    Aggregates normalized volume per quantization bin.
    """
    B = edges.size - 1

    @pandas_udf(ArrayType(DoubleType()), PandasUDFType.SCALAR)
    def bin(std_prices: pd.Series, norm_volumes: pd.Series) -> pd.Series:
        out = []

        for std_price, norm_volume in zip(std_prices, norm_volumes):
            std_price = np.asarray(std_price, dtype=float)
            norm_volume = np.asarray(norm_volume, dtype=float)
            n = min(len(std_price), len(norm_volume))
            std_price, norm_volume = std_price[:n], norm_volume[:n]

            # Drops NaN values
            mask = np.isfinite(std_price) & np.isfinite(norm_volume)
            std_price, norm_volume = std_price[mask], norm_volume[mask]
            
            # Assigns bin indices
            idx = np.digitize(std_price, edges, right=False) - 1
            idx = np.clip(idx, 0, B - 1)

            # Aggregates volumes into bins
            bins = np.zeros(B, dtype=float)
            np.add.at(bins, idx, norm_volume)
            out.append(bins.tolist())
        
        return pd.Series(out)
    
    return lob.withColumn("bins", bin(col("prices"), col("volumes")))

def split_bins(lob: DataFrame, edges: np.ndarray) -> DataFrame:
    """
    Splits volume bins into negative and positive quantized standarized relative price related to these bins.
    """
    centers = (edges[:-1] + edges[1:]) / 2
    neg_idx = np.where(centers < 0)[0].tolist()
    pos_idx = np.where(centers > 0)[0].tolist()

    schema = StructType(
        [
            StructField("neg_bins", ArrayType(DoubleType()), True),
            StructField("pos_bins", ArrayType(DoubleType()), True),
        ]
    )

    @pandas_udf(schema, PandasUDFType.SCALAR)
    def split(bins: pd.Series) -> pd.DataFrame:
        neg, pos = [], []

        for arr in bins:
            arr = np.array(arr, dtype=float)
            neg.append(arr[neg_idx].tolist())
            pos.append(arr[pos_idx].tolist())

        return pd.DataFrame({"neg_bins": neg, "pos_bins": pos})
    
    res = split(col("bins"))

    return lob.withColumn("neg_bins", res["neg_bins"]).withColumn("pos_bins", res["pos_bins"])

def bin_lob(lob: DataFrame) -> DataFrame:
    """
    Quantizes LOB snapshot standarized relative prices and bins normalized volumes.
    """
    best_delta, binning_edges, best_snr = quantize_lob(
        lob, L=10.0, B=1000, w_factor=0.1, delta_grid=np.logspace(-4, 2, 50), clip_max_percentile=0.999
    )
    lob = aggregate_volume_per_bin(lob, binning_edges)
    lob = split_bins(lob, binning_edges)

    return lob.select("timestamp", "neg_bins", "pos_bins")

# ========================================================================================================== #
# LOB Feature Derivation                                                                                     #
# ========================================================================================================== #

def calculate_spread(lob: DataFrame) -> DataFrame:
    """
    Calculates bid-ask spread from best bid and best ask prices.
    """
    best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
    best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")

    return lob.withColumn("spread", best_ask - best_bid)

def calculate_depth_imbalance(lob: DataFrame, k: int) -> DataFrame:
    """
    Calculates depth imbalance.
    
    If k > 0, uses top-k bid / ask levels (highest k bids, lowest k asks).
    If k <= 0, uses all available levels.
    """
    K = int(k)
    eps = 1e-18

    @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
    def depth_imbalance(bids: pd.Series, asks: pd.Series) -> pd.Series:
        out = []

        for bids, asks in zip(bids, asks):
            bids_arr = np.asarray(bids if bids is not None else [], dtype=object)
            asks_arr = np.asarray(asks if asks is not None else [], dtype=object)

            vb, va = 0.0, 0.0

            if bids_arr.size > 0:
                bids_mat = np.array(bids_arr.tolist(), dtype=float).reshape(-1, 2)
                bids_sorted = bids_mat[np.argsort(-bids_mat[:, 0])]
                vb = bids_sorted[:K, 1].sum() if K > 0 else bids_sorted[:, 1].sum()

            if asks_arr.size > 0:
                asks_mat = np.array(asks_arr.tolist(), dtype=float).reshape(-1, 2)
                asks_sorted = asks_mat[np.argsort(asks_mat[:, 0])]
                va = asks_sorted[:K, 1].sum() if K > 0 else asks_sorted[:, 1].sum()

            denom = vb + va
            imb = 0.0 if denom <= eps else (vb - va) / denom
            out.append(float(imb))

        return pd.Series(out, dtype="float64")

    suffix = f"top_{K}" if K > 0 else "all"

    return lob.withColumn(f"depth_imbalance_{suffix}", depth_imbalance(col("bids"), col("asks")))

def calculate_order_flow_imbalance(lob: DataFrame, k: int) -> DataFrame:
    """
    Calculates order flow imbalance.

    If k > 0, uses top-k bid / ask levels (highest k bids, lowest k asks).
    If k <= 0, uses all available levels.
    """
    K = int(k)

    # Adds previous snapshot bids and asks
    w = Window.orderBy("timestamp")

    lob_prev = (
        lob
        .withColumn("bids_prev", lag(col("bids"), 1).over(w))
        .withColumn("asks_prev", lag(col("asks"), 1).over(w))
    )

    @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
    def order_flow_imbalance(bids_t_s: pd.Series, asks_t_s: pd.Series, bids_p_s: pd.Series, asks_p_s: pd.Series) -> pd.Series:

        def top_k(arr, side, K):
            """
            Extracts top-k levels from a bid/ask array and sorts them by price.
            """
            if arr is None:
                m = np.empty((0, 2), dtype=float)

            else:
                a = np.asarray(arr, dtype=object)

                if a.size == 0:
                    m = np.empty((0, 2), dtype=float)

                else:
                    m = np.array(a.tolist(), dtype=float).reshape(-1, 2)
                    m = m[np.argsort(-m[:, 0])] if side == "bid" else m[np.argsort(m[:, 0])]

            if K > 0:
                m = m[:K]

            return m

        def pad_pair(curr, prev):
            """
            Pads the shorter array of a bid/ask pair to align price levels.
            """
            n = max(len(curr), len(prev))

            if len(curr) < n:
                pad = np.zeros((n - len(curr), 2), dtype=float)
                src = prev[len(curr):, 0] if len(prev) > len(curr) else (curr[-1:, 0] if len(curr) else np.array([0.0]))
                pad[:, 0] = src if src.shape == (n - len(curr),) else np.repeat(src[0], n - len(curr))
                curr = np.vstack([curr, pad])

            if len(prev) < n:
                pad = np.zeros((n - len(prev), 2), dtype=float)
                src = curr[len(prev):, 0] if len(curr) > len(prev) else (prev[-1:, 0] if len(prev) else np.array([0.0]))
                pad[:, 0] = src if src.shape == (n - len(prev),) else np.repeat(src[0], n - len(prev))
                prev = np.vstack([prev, pad])

            return curr, prev

        out = []

        for bids_t, asks_t, bids_p, asks_p in zip(bids_t_s, asks_t_s, bids_p_s, asks_p_s):
            # Extracts and sorts top-k levels
            bt = top_k(bids_t, "bid", K)
            at = top_k(asks_t, "ask", K)
            bp = top_k(bids_p, "bid", K)
            ap = top_k(asks_p, "ask", K)

            # Pads to align price levels
            bt, bp = pad_pair(bt, bp)
            at, ap = pad_pair(at, ap)

            # Calculates bids contribution
            bid_up_or_same = (bt[:, 0] >= bp[:, 0]).astype(float)
            bid_down_or_same = (bt[:, 0] <= bp[:, 0]).astype(float)
            e_b = (bid_up_or_same * bt[:, 1] - bid_down_or_same * bp[:, 1]).sum()

            # Calculates asks contribution 
            ask_down_or_same = (at[:, 0] <= ap[:, 0]).astype(float)
            ask_up_or_same   = (at[:, 0] >= ap[:, 0]).astype(float)
            e_a = (ask_down_or_same * at[:, 1] - ask_up_or_same * ap[:, 1]).sum()

            out.append(float(e_b - e_a))

        return pd.Series(out, dtype="float64")

    suffix = f"top_{K}" if K > 0 else "all"
    colname = f"ofi_{suffix}"

    res = lob_prev.withColumn(colname, order_flow_imbalance(col("bids"), col("asks"), col("bids_prev"), col("asks_prev")))
    res = res.drop("bids_prev", "asks_prev")

    return res

def calculate_microprice(lob: DataFrame) -> DataFrame:
    """
    Computes standard microprice.
    """
    best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
    best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")

    lob = (
        lob
        .withColumn("best_bid", best_bid)
        .withColumn("best_ask", best_ask)
    )

    best_bid_vol = expr("aggregate(filter(bids, x -> x[0] = best_bid_price), 0D, (acc,x) -> acc + x[1])")
    best_ask_vol = expr("aggregate(filter(asks, x -> x[0] = best_ask_price), 0D, (acc,x) -> acc + x[1])")

    lob = (
        lob
        .withColumn("best_bid_vol", best_bid_vol)
        .withColumn("best_ask_vol", best_ask_vol)
    )

    # Calculates microprice with safe fallback to mid-price when the denominator is zero
    micro_expr = when(
        (col("best_bid_vol") + col("best_ask_vol")) == 0.0,
        (col("best_bid") + col("best_ask")) / 2.0
    ).otherwise(
        (col("best_ask") * col("best_bid_vol") + col("best_bid") * col("best_ask_vol")) /
        (col("best_bid_vol") + col("best_ask_vol"))
    )

    lob = lob.withColumn("microprice", micro_expr)
    lob = lob.drop("best_bid_price", "best_ask_price", "best_bid_vol", "best_ask_vol")

    return lob

def calculate_midprice_volatility(lob: DataFrame) -> DataFrame:
    """
    Calculates mid-price volatility, defined as the square root of mid-price log-returns variance proxy.
    """
    return lob.withColumn("midprice_vol", expr("sqrt(variance_proxy)"))

def calculate_market_depth(lob: DataFrame, k: int) -> DataFrame:
    """
    Calculates total market depth.

    If k > 0, uses top-k bid / ask levels (highest k bids, lowest k asks).
    If k <= 0, uses all available levels.
    """
    K = int(k)

    @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
    def depth(bids: pd.Series, asks: pd.Series) -> pd.Series:
        out = []

        for bid, ask in zip(bids, asks):
            if bid is not None and len(bid) > 0:
                b = np.array(bid, dtype=object)
                b = np.array(b.tolist(), dtype=float).reshape(-1, 2)
                b = b[np.argsort(-b[:, 0])]

            else:
                b = np.empty((0, 2), dtype=float)

            if ask is not None and len(ask) > 0:
                a = np.array(ask, dtype=object)
                a = np.array(a.tolist(), dtype=float).reshape(-1, 2)
                a = a[np.argsort(a[:, 0])]

            else:
                a = np.empty((0, 2), dtype=float)

            if K > 0:
                b = b[:K]
                a = a[:K]

            # Calculates total depth
            depth_sum = float(b[:, 1].sum() + a[:, 1].sum())
            out.append(depth_sum)

        return pd.Series(out, dtype="float64")

    suffix = f"top_{K}" if K > 0 else "all"
    colname = f"depth_{suffix}"

    return lob.withColumn(colname, depth(col("bids"), col("asks")))

def calculate_lob_slope(lob: DataFrame, k: int, eps: float = 1e-12) -> DataFrame:
    """
    Calculates LOB slope.

    If k > 0, uses top-k bid / ask levels (highest k bids, lowest k asks).
    If k <= 0, uses all available levels.
    """
    K = int(k)

    @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
    def slope(bids: pd.Series, asks: pd.Series) -> pd.Series:

        def to_mat(arr, side):
            """
            Converts bids and asks array to a sorted numpy matrix.
            """
            if arr is None or len(arr) == 0:
                m = np.empty((0, 2), dtype=float)

            else:
                a = np.array(arr, dtype=object)
                m = np.array(a.tolist(), dtype=float).reshape(-1, 2)
                m = m[np.argsort(-m[:, 0])] if side == "bid" else m[np.argsort(m[:, 0])]

            return m

        out = []

        for bid, ask in zip(bids, asks):
            b = to_mat(bid, "bid")
            a = to_mat(ask, "ask")

            # Extracts best bid and best ask
            if b.shape[0] > 0:
                best_bid = b[0, 0]

            else:
                best_bid = np.nan

            if a.shape[0] > 0:
                best_ask = a[0, 0]

            else:
                best_ask = np.nan

            if not np.isfinite(best_bid) or not np.isfinite(best_ask):
                out.append(np.nan)
                continue

            mid = 0.5 * (best_bid + best_ask)

            # Uses top-K price levels
            b_use = b[:K] if K > 0 else b
            a_use = a[:K] if K > 0 else a

            # Calculates distances to mid-price
            db = mid - b_use[:, 0] if b_use.size else np.empty(0)
            da = a_use[:, 0] - mid if a_use.size else np.empty(0)

            if b_use.size:
                mask_b = db > 0
                vb = b_use[mask_b, 1]
                db = db[mask_b]

            else:
                vb = np.empty(0); db = np.empty(0)

            if a_use.size:
                mask_a = da > 0
                va = a_use[mask_a, 1]
                da = da[mask_a]

            else:
                va = np.empty(0); da = np.empty(0)

            # Calculates densities
            Db = float(np.sum(vb / (db + eps))) if db.size else 0.0
            Da = float(np.sum(va / (da + eps))) if da.size else 0.0

            # Calculates LOB slope
            slope = (Da + eps) / (Db + eps)
            out.append(float(slope))

        return pd.Series(out, dtype="float64")

    suffix = f"top_{K}" if K > 0 else "all"
    colname = f"lob_slope_{suffix}"

    return lob.withColumn(colname, slope(col("bids"), col("asks")))

def calculate_liquidity_concentration(lob: DataFrame, k: int) -> DataFrame:
    """
    Calculates liquidity concentration .
    
    If k > 0, uses top-k bid / ask levels (highest k bids, lowest k asks).
    If k <= 0, uses all available levels.
    """
    K = int(k)

    @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
    def liquidity_concentration(bids: pd.Series, asks: pd.Series) -> pd.Series:
        out = []

        for bid, ask in zip(bids, asks):
            if bid is not None and len(bid) > 0:
                b = np.array(bid, dtype=object)
                b = np.array(b.tolist(), dtype=float).reshape(-1, 2)
                b = b[np.argsort(-b[:, 0])]

            else:
                b = np.empty((0, 2), dtype=float)

            if ask is not None and len(ask) > 0:
                a = np.array(ask, dtype=object)
                a = np.array(a.tolist(), dtype=float).reshape(-1, 2)
                a = a[np.argsort(a[:, 0])]

            else:
                a = np.empty((0, 2), dtype=float)

            # Denominator: all levels
            denom = float(b[:, 1].sum() + a[:, 1].sum())

            # Numerator: top-K (or all if K <= 0)
            if K > 0:
                num = float(b[:K, 1].sum() + a[:K, 1].sum())

            else:
                num = denom

            # Calculates liquidity concentration
            lc = 0.0 if denom <= 0.0 else (num / denom)
            out.append(lc)

        return pd.Series(out, dtype="float64")

    suffix = f"top_{K}" if K > 0 else "all"
    colname = f"liquidity_concentration_{suffix}"

    return lob.withColumn(colname, liquidity_concentration(col("bids"), col("asks")))

def calculate_price_impact_proxy(lob: DataFrame, k: int, eps: float = 1e-12) -> DataFrame:
    """
    Kyle-like price impact proxy (lambda).

    If k > 0, uses top-k bid / ask levels (highest k bids, lowest k asks).
    If k <= 0, uses all available levels.
    """
    K = int(k)

    @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
    def price_impact_proxy(bids: pd.Series, asks: pd.Series) -> pd.Series:

        def to_sorted(mat, side):
            if mat is None or len(mat) == 0:
                return np.empty((0, 2), dtype=float)
            
            a = np.array(mat, dtype=object)
            m = np.array(a.tolist(), dtype=float).reshape(-1, 2)

            return m[np.argsort(-m[:, 0])] if side == "bid" else m[np.argsort(m[:, 0])]

        out = []

        for bid, ask in zip(bids, asks):
            b = to_sorted(bid, "bid")
            a = to_sorted(ask, "ask")

            if b.shape[0] == 0 or a.shape[0] == 0:
                out.append(np.nan)
                continue

            # Determines effective K for each side
            Kb = b.shape[0] if K <= 0 else min(K, b.shape[0])
            Ka = a.shape[0] if K <= 0 else min(K, a.shape[0])

            # Prices at level K
            PbK = b[Kb - 1, 0]
            PaK = a[Ka - 1, 0]

            # Calculates cumulative depth up to K
            depth_sum = float(b[:Kb, 1].sum() + a[:Ka, 1].sum())

            # Calculates price impact proxy
            lam = 0.0 if depth_sum <= 0.0 else (PaK - PbK) / max(depth_sum, eps)
            out.append(float(lam))

        return pd.Series(out, dtype="float64")

    suffix = f"top_{K}" if K > 0 else "all"
    colname = f"price_impact_proxy_{suffix}"

    return lob.withColumn(colname, price_impact_proxy(col("bids"), col("asks")))

def calculate_liquidity_spread(lob: DataFrame, k: int, eps: float = 1e-12) -> DataFrame:
    """
    Calculate liquidity spread 

    If k > 0, uses top-k bid / ask levels (highest k bids, lowest k asks).
    If k <= 0, uses all available levels.
    """
    K = int(k)

    @pandas_udf(DoubleType(), PandasUDFType.SCALAR)
    def liquidity_spread(bids: pd.Series, asks: pd.Series) -> pd.Series:

        def to_sorted(mat, side):
            if mat is None or len(mat) == 0:
                return np.empty((0, 2), dtype=float)
            
            a = np.array(mat, dtype=object)
            m = np.array(a.tolist(), dtype=float).reshape(-1, 2)

            return m[np.argsort(-m[:, 0])] if side == "bid" else m[np.argsort(m[:, 0])]

        out = []

        for bid, ask in zip(bids, asks):
            b = to_sorted(bid, "bid")
            a = to_sorted(ask, "ask")

            if b.shape[0] == 0 or a.shape[0] == 0:
                out.append(0.0)
                continue

            Kb, Ka = b.shape[0], a.shape[0]
            K_eff = min(Kb, Ka) if K <= 0 else min(K, Kb, Ka)

            if K_eff <= 0:
                out.append(0.0)
                continue

            Pb = b[:K_eff, 0]
            Vb = b[:K_eff, 1]
            Pa = a[:K_eff, 0]
            Va = a[:K_eff, 1]

            vol_sum = Va + Vb
            denom = float(np.sum(vol_sum))

            if denom <= 0.0:
                out.append(0.0)
                continue

            num = float(np.sum((Pa - Pb) * vol_sum))
            lspread = num / max(denom, eps)
            out.append(lspread)

        return pd.Series(out, dtype="float64")

    suffix = f"top_{K}" if K > 0 else "all"
    colname = f"liquidity_spread_{suffix}"

    return lob.withColumn(colname, liquidity_spread(col("bids"), col("asks")))

if __name__ == "__main__":
    lob = load_lob()

    # LOB Feature Derivation
    lob = calculate_mid_prices(lob)

    # LOB standardization and Binnarization
    # std_lob = standarize_lob(lob)
    # binn_lob = bin_lob(std_lob)
    # save_lob(binn_lob)

    # Trade Pre-processing
    # trades = load_trade()
    # save_trade(trades)

    spark.stop()