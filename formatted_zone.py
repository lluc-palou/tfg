import numpy as np
import pandas as pd
from typing import Sequence, Tuple
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType, TimestampType
from pyspark.sql.functions import col, expr, from_json, regexp_replace, log, lag, pandas_udf, lit, PandasUDFType, concat

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
LANDING_TRADE = "landing_trade_summary"
LANDING_LOB = "landing_lob"
FORMATTED_TRADE = "formatted_trade_summary"
FORMATTED_LOB = "formatted_lob"
LOG_COLL = "log"

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
    
    # Parses data and casts data types
    lob = lob \
        .withColumn("timestamp", col("timestamp").cast("timestamp")) \
        .withColumn("bids", from_json(regexp_replace(col("bids").cast("string"), r"'", '"'), ArrayType(ArrayType(DoubleType())))) \
        .withColumn("asks", from_json(regexp_replace(col("asks").cast("string"), r"'", '"'), ArrayType(ArrayType(DoubleType())))) \
        .orderBy("timestamp").limit(50) # Remove the limit function in production

    return lob

def save_lob(lob: DataFrame) -> None:
    """
    Saves pre-processed LOB dataframe into Formatted Zone overwritting the target collection.
    """
    lob.write.format("mongodb") \
        .option("database", DB_NAME) \
        .option("collection", FORMATTED_LOB) \
        .mode("overwrite") \
        .save()

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

if __name__ == "__main__":
    lob = load_lob()
    std_lob = standarize_lob(lob)
    binn_lob = bin_lob(std_lob)
    save_lob(binn_lob)
    spark.stop()