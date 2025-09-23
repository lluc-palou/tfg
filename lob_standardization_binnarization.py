import math
import sys, time
from datetime import datetime
import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import LongType, DoubleType, ArrayType, StructType, StructField
from pyspark.sql.functions import (
    col, expr, log, lag, when, to_date, pow, row_number, lit, sum as spark_sum, 
    date_trunc, approx_count_distinct, udf, collect_list
)

# ==============================
# Spark & Mongo configuration
# ==============================

jar_files_path = "file:///C:/Users/llucp/spark_jars/"

jar_files = [
    "mongo-spark-connector_2.12-10.1.1.jar",
    "mongodb-driver-core-4.10.1.jar",
    "mongodb-driver-sync-4.10.1.jar",
    "bson-4.10.1.jar"
]

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "tfg"
LANDING_LOB = "landing_lob"
FORMATTED_LOB = "formatting_lob"

spark = (
    SparkSession.builder
    .appName("StandardizedZone")
    .config("spark.jars", ",".join([jar_files_path + jar for jar in jar_files]))

    # Arrow + AQE etc.
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "512")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.shuffle.partitions", "16")

    # Mongo
    .config("spark.mongodb.read.connection.uri", MONGO_URI)
    .config("spark.mongodb.write.connection.uri", MONGO_URI)
    .config("spark.mongodb.read.database", DB_NAME)
    .config("spark.mongodb.write.database", DB_NAME)

    # Keep bulks tiny
    .config("spark.mongodb.write.ordered", "false")
    .config("spark.mongodb.write.writeConcern.w", "1")

    # Timeouts / robustness
    .config("spark.mongodb.connection.timeout.ms", "30000")
    .config("spark.mongodb.socket.timeout.ms", "120000")
    .config("spark.mongodb.write.retryWrites", "true")

    # Memory / network tolerance
    .config("spark.driver.memory", "8g")
    .config("spark.network.timeout", "300s")
    .config("spark.executor.heartbeatInterval", "60s")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir("file:///C:/tmp/spark_checkpoints")

# ==============================
# IO Functions
# ==============================

def logger(msg: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def load_lob() -> DataFrame:
    """
    Loads Landing LOB with projection pushdown (timestamp/bids/asks only).
    Same as feature extraction script.
    """
    pipeline = [
        {"$project": {"timestamp": 1, "bids": 1, "asks": 1}}
    ]
    lob = (
        spark.read.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", LANDING_LOB)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )
    return lob

def get_hourly_ranges(df: DataFrame) -> list:
    """
    Get list of distinct hour ranges in the dataset for batch processing.
    """
    hours = (df.withColumn("trading_hour", date_trunc("hour", col("timestamp")))
               .select("trading_hour")
               .distinct()
               .orderBy("trading_hour")
               .collect())
    return [row.trading_hour for row in hours]

def load_hour_batch(df: DataFrame, hour_start, include_overlap: bool = True):
    """
    Load data for a specific hour batch.
    If include_overlap=True, includes some data from previous hour for continuity.
    """
    if include_overlap:
        # Include 10 minutes from previous hour for continuity
        prev_hour_start = expr(f"timestamp >= '{hour_start}' - INTERVAL 10 MINUTES")
        current_hour = expr(f"timestamp < '{hour_start}' + INTERVAL 1 HOUR")
        filter_condition = prev_hour_start & current_hour
    else:
        # Exact hour only
        filter_condition = (
            (col("timestamp") >= lit(hour_start)) & 
            (col("timestamp") < expr(f"'{hour_start}' + INTERVAL 1 HOUR"))
        )
    
    return df.filter(filter_condition)

def write_append_mongo(df: DataFrame) -> None:
    (df.write.format("mongodb")
       .option("database", DB_NAME)
       .option("collection", FORMATTED_LOB)
       .option("ordered", "false")
       .mode("append")
       .save())

# ==============================
# Feature Derivation (from feature extraction script)
# ==============================

def calculate_mid_prices(df: DataFrame) -> DataFrame:
    """Same as feature extraction script"""
    best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
    best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")
    return df.withColumn("mid_price", (best_bid + best_ask) / 2.0)

def calculate_log_returns(df: DataFrame) -> DataFrame:
    """
    BATCH-LOCAL log-returns ordered by timestamp within the batch.
    Same as feature extraction script.
    """
    w = Window.orderBy("timestamp")
    return df.withColumn("log_return", log(col("mid_price")) - log(lag(col("mid_price"), 1).over(w)))

def estimate_variance(df: DataFrame, half_life: int = 20) -> DataFrame:
    """
    BATCH-LOCAL SQL-only EWMA of squared log-returns.
    Same as feature extraction script.
    """
    alpha = 1.0 - pow(lit(2.0), lit(-1.0) / lit(float(half_life)))
    beta  = (lit(1.0) - alpha)

    w = Window.orderBy("timestamp")
    rn = row_number().over(w) - lit(1)

    df = df.withColumn("r2", col("log_return") * col("log_return"))
    df = df.withColumn("z_i", when(col("r2").isNotNull(), col("r2") * pow(beta, -rn)).otherwise(lit(None)))

    wcum = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df = df.withColumn("cum_z", spark_sum("z_i").over(wcum))

    df = df.withColumn("variance_proxy", alpha * pow(beta, rn) * col("cum_z"))
    df = df.drop("r2", "z_i", "cum_z")
    return df

def derive_required_features(hour_batch: DataFrame) -> DataFrame:
    """
    Derives only the required features for standardization.
    Following the same pattern as feature extraction script.
    """
    logger('derive_required_features: START')
    
    base = (
        hour_batch.select("timestamp","bids","asks")
                  .withColumn("trading_day", to_date(col("timestamp")))
                  .withColumn("trading_hour", date_trunc("hour", col("timestamp")))
                  .repartition(4)  # smaller partitions for hour batches
                  .sortWithinPartitions("timestamp")
                  .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = base.count()  # materialize once
    logger('derive_required_features: base materialized')

    logger('derive_required_features: mid_price...')
    df = calculate_mid_prices(base).select("timestamp","trading_day","trading_hour","bids","asks","mid_price")
    
    logger('derive_required_features: log_return...')
    df = calculate_log_returns(df).select("timestamp","trading_day","trading_hour","bids","asks","mid_price","log_return")
    
    logger('derive_required_features: variance...')
    df = estimate_variance(df, half_life=20).select("timestamp","trading_day","trading_hour","bids","asks",
                                                    "mid_price","log_return","variance_proxy")

    cnt = df.count()
    logger(f'derive_required_features: DONE features, rows={cnt}')
    base.unpersist()
    return df

# ==============================
# LOB Standardization (SQL-optimized)
# ==============================

def standardize_prices(df: DataFrame) -> DataFrame:
    """
    Standardizes bid and ask prices by subtracting mid-prices and dividing by volatility times the mid-prices.
    SQL-optimized version using Spark expressions.
    """
    eps = 1e-12
    
    # Standardize bids and asks using SQL expressions
    df = df.withColumn(
        "bids", 
        expr(f"transform(bids, x -> array((x[0] - mid_price) / (sqrt(variance_proxy) * mid_price + {eps}), x[1]))")
    ).withColumn(
        "asks", 
        expr(f"transform(asks, x -> array((x[0] - mid_price) / (sqrt(variance_proxy) * mid_price + {eps}), x[1]))")
    )
    
    return df

def normalize_volumes(df: DataFrame) -> DataFrame:
    """
    Normalizes trading volume using SQL expressions for efficiency.
    """
    # Calculate total volume in one expression
    df = df.withColumn(
        "total_volume",
        expr("aggregate(bids, 0D, (acc, x) -> acc + x[1]) + aggregate(asks, 0D, (acc, x) -> acc + x[1])")
    )

    # Normalize volumes
    df = df.withColumn(
        "bids", 
        expr("transform(bids, x -> array(x[0], CASE WHEN total_volume = 0 THEN 0D ELSE x[1] / total_volume END))")
    ).withColumn(
        "asks", 
        expr("transform(asks, x -> array(x[0], CASE WHEN total_volume = 0 THEN 0D ELSE x[1] / total_volume END))")
    )

    return df.drop("total_volume")

def flatten_bids_and_asks(df: DataFrame) -> DataFrame:
    """
    Flattens bid and ask standardized and normalized volume pairs using SQL.
    """
    df = df.withColumn("pairs", expr("concat(bids, asks)"))
    df = df.withColumn("prices", expr("transform(pairs, x -> x[0])"))
    df = df.withColumn("volumes", expr("transform(pairs, x -> x[1])"))

    return df.drop("pairs")

def calculate_gamma_scaling_factor_sql(df: DataFrame, L: float = 10.0, target: float = 0.95) -> DataFrame:
    """
    SQL-optimized gamma calculation using broadcast variables and vectorized operations.
    """
    gammas_grid = np.linspace(100, 5000, 80)
    
    # Broadcast gamma grid for efficiency
    gamma_grid_broadcast = spark.sparkContext.broadcast(gammas_grid.tolist())
    
    @udf(DoubleType())
    def calculate_gamma_udf(prices_arr, volumes_arr):
        if not prices_arr or not volumes_arr:
            return 1000.0  # default gamma
            
        prices = np.array(prices_arr, dtype=np.float64)
        volumes = np.array(volumes_arr, dtype=np.float64)
        
        # Handle invalid data
        valid_mask = np.isfinite(prices) & np.isfinite(volumes) & (volumes >= 0)
        if not np.any(valid_mask):
            return 1000.0
            
        prices = prices[valid_mask]
        volumes = volumes[valid_mask]
        
        if len(prices) == 0:
            return 1000.0
        
        best_gamma = 1000.0  # default
        best_diff = float("inf")
        
        for gamma in gamma_grid_broadcast.value:
            # Calculate coverage
            coverage_mask = np.abs(prices / gamma) <= L
            coverage = volumes[coverage_mask].sum() if np.any(coverage_mask) else 0.0
            diff = abs(coverage - target)
            
            if diff < best_diff:
                best_diff = diff
                best_gamma = gamma
        
        return float(best_gamma)
    
    df = df.withColumn("gamma", calculate_gamma_udf(col("prices"), col("volumes")))
    return df

def gamma_scale_std_prices(df: DataFrame) -> DataFrame:
    """
    Applies gamma scaling to standardized prices using SQL.
    """
    return df.withColumn("prices", expr("transform(prices, x -> x / gamma)"))

def standardize_lob_batch(df: DataFrame) -> DataFrame:
    """
    Pipeline for standardizing LOB data in batch processing.
    SQL-optimized version of the standardization pipeline.
    """
    logger('standardize_lob_batch: START')
    
    logger('standardize_lob_batch: standardize_prices...')
    df = standardize_prices(df)
    
    logger('standardize_lob_batch: normalize_volumes...')
    df = normalize_volumes(df)
    
    logger('standardize_lob_batch: flatten_bids_and_asks...')
    df = flatten_bids_and_asks(df)
    
    logger('standardize_lob_batch: calculate_gamma_scaling_factor...')
    df = calculate_gamma_scaling_factor_sql(df, L=10.0, target=0.95)
    
    logger('standardize_lob_batch: gamma_scale_std_prices...')
    df = gamma_scale_std_prices(df)
    
    logger('standardize_lob_batch: DONE')
    return df

# ==============================
# LOB Binnarization (SQL-optimized) 
# ==============================

def quantize_lob_batch(df: DataFrame, L: float = 10.0, B: int = 1000, w_factor: float = 0.1, 
                      clip_max_percentile: float = 0.999) -> tuple:
    """
    SQL-optimized quantization that processes data in batch.
    Returns optimal delta, edges, and SNR.
    """
    logger('quantize_lob_batch: collecting data for optimization...')
    
    # Collect data efficiently using SQL aggregations
    data_rows = df.select("prices", "volumes", "gamma").collect()
    
    all_prices = []
    all_volumes = []
    gammas = []
    
    for row in data_rows:
        if row["prices"] and row["volumes"]:
            prices = np.array(row["prices"], dtype=np.float64)
            volumes = np.array(row["volumes"], dtype=np.float64)
            gamma = row["gamma"]
            
            # Filter valid data
            valid_mask = np.isfinite(prices) & np.isfinite(volumes) & (volumes >= 0)
            if np.any(valid_mask):
                all_prices.append(prices[valid_mask])
                all_volumes.append(volumes[valid_mask])
                gammas.append(gamma)
    
    if not all_prices:
        logger('quantize_lob_batch: No valid data found, using default parameters')
        # Return default parameters
        edges = np.linspace(-L, L, B + 1)
        return 0.1, edges, 1.0
    
    all_prices = np.concatenate(all_prices)
    all_volumes = np.concatenate(all_volumes)
    
    # Calculate reference gamma and parameters
    gamma_ref = np.median(gammas) if gammas else 1000.0
    w0 = w_factor / gamma_ref
    
    # Global volume normalization
    volume_sum = all_volumes.sum()
    if volume_sum > 0:
        all_volumes = all_volumes / volume_sum
    
    # Calculate price threshold
    abs_prices = np.abs(all_prices)
    
    def weighted_quantile(x, w, p):
        if len(x) == 0:
            return L
        order = np.argsort(x)
        xx, ww = x[order], w[order]
        cdf = np.cumsum(ww)
        cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
        idx = np.searchsorted(cdf, p, side="left")
        return float(xx[min(idx, len(xx) - 1)])
    
    prices_max = weighted_quantile(abs_prices, all_volumes, clip_max_percentile)
    
    # Define edge calculation function
    bins_per_side = (B - 4) // 2
    
    def calculate_edges(delta):
        y_max = np.log1p((L - w0) / delta)
        y_edges = np.linspace(0.0, y_max, bins_per_side + 1, dtype=np.float64)
        x_pos_edges = w0 + delta * np.expm1(y_edges)
        x_neg_edges = -x_pos_edges[::-1]
        
        edges = np.concatenate([
            np.array([-prices_max]),
            x_neg_edges[:-1],
            np.array([-w0, 0.0, w0]),
            x_pos_edges[1:],
            np.array([prices_max])
        ])
        
        return edges
    
    # Optimize delta for maximum SNR
    delta_grid = np.logspace(-4, 2, 50)
    eps = 1e-18
    
    def calculate_snr(delta):
        edges = calculate_edges(delta)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        idx = np.digitize(all_prices, edges, right=False) - 1
        idx = np.clip(idx, 0, B - 1)
        quantized_prices = bin_centers[idx]
        
        signal = float(np.sum(all_volumes * (all_prices ** 2)))
        mse = float(np.sum(all_volumes * ((all_prices - quantized_prices) ** 2)))
        snr = signal / max(mse, eps)
        
        return snr, edges
    
    logger('quantize_lob_batch: optimizing delta...')
    best_snr, best_delta, best_edges = -np.inf, 0.1, None
    
    for delta in delta_grid:
        snr, edges = calculate_snr(delta)
        if snr > best_snr:
            best_snr, best_delta, best_edges = snr, delta, edges
    
    logger(f'quantize_lob_batch: optimal delta={best_delta:.6f}, SNR={best_snr:.6f}')
    return best_delta, best_edges, best_snr

def aggregate_volume_per_bin_sql(df: DataFrame, edges: np.ndarray) -> DataFrame:
    """
    SQL-optimized volume aggregation per bin using UDF with broadcast edges.
    """
    B = len(edges) - 1
    edges_broadcast = spark.sparkContext.broadcast(edges.tolist())
    
    @udf(ArrayType(DoubleType()))
    def bin_volumes_udf(prices_arr, volumes_arr):
        if not prices_arr or not volumes_arr:
            return [0.0] * B
            
        prices = np.array(prices_arr, dtype=np.float64)
        volumes = np.array(volumes_arr, dtype=np.float64)
        
        # Handle edge cases
        n = min(len(prices), len(volumes))
        if n == 0:
            return [0.0] * B
            
        prices = prices[:n]
        volumes = volumes[:n]
        
        # Filter valid data
        valid_mask = np.isfinite(prices) & np.isfinite(volumes)
        prices = prices[valid_mask]
        volumes = volumes[valid_mask]
        
        if len(prices) == 0:
            return [0.0] * B
        
        # Bin assignment
        edges_arr = np.array(edges_broadcast.value)
        idx = np.digitize(prices, edges_arr, right=False) - 1
        idx = np.clip(idx, 0, B - 1)
        
        # Aggregate volumes
        bins = np.zeros(B, dtype=np.float64)
        np.add.at(bins, idx, volumes)
        
        return bins.tolist()
    
    return df.withColumn("bins", bin_volumes_udf(col("prices"), col("volumes")))

def split_bins_sql(df: DataFrame, edges: np.ndarray) -> DataFrame:
    """
    SQL-optimized bin splitting using broadcast variables.
    """
    centers = (edges[:-1] + edges[1:]) / 2
    neg_indices = np.where(centers < 0)[0].tolist()
    pos_indices = np.where(centers > 0)[0].tolist()
    
    # Broadcast indices for efficiency
    neg_idx_broadcast = spark.sparkContext.broadcast(neg_indices)
    pos_idx_broadcast = spark.sparkContext.broadcast(pos_indices)
    
    @udf(ArrayType(DoubleType()))
    def extract_neg_bins(bins_arr):
        if not bins_arr:
            return [0.0] * len(neg_idx_broadcast.value)
        bins = np.array(bins_arr, dtype=np.float64)
        return bins[neg_idx_broadcast.value].tolist()
    
    @udf(ArrayType(DoubleType()))
    def extract_pos_bins(bins_arr):
        if not bins_arr:
            return [0.0] * len(pos_idx_broadcast.value)
        bins = np.array(bins_arr, dtype=np.float64)
        return bins[pos_idx_broadcast.value].tolist()
    
    df = df.withColumn("neg_bins", extract_neg_bins(col("bins")))
    df = df.withColumn("pos_bins", extract_pos_bins(col("bins")))
    
    return df.drop("bins")

def binnarize_lob_batch(df: DataFrame) -> DataFrame:
    """
    Pipeline for binnarizing LOB data in batch processing.
    SQL-optimized version of the binnarization pipeline.
    """
    logger('binnarize_lob_batch: START')
    
    # Quantize to get optimal parameters
    best_delta, edges, best_snr = quantize_lob_batch(
        df, L=10.0, B=1000, w_factor=0.1, clip_max_percentile=0.999
    )
    
    logger('binnarize_lob_batch: aggregate_volume_per_bin...')
    df = aggregate_volume_per_bin_sql(df, edges)
    
    logger('binnarize_lob_batch: split_bins...')
    df = split_bins_sql(df, edges)
    
    logger('binnarize_lob_batch: DONE')
    return df.select("timestamp", "trading_hour", "neg_bins", "pos_bins")

# ==============================
# Batch Processing Pipeline
# ==============================

def process_hour_standardization(hour_batch: DataFrame) -> DataFrame:
    """
    Process standardization and binnarization for a single hour batch.
    Following the same workflow as feature extraction.
    """
    logger('process_hour_standardization: START')
    
    # 1) First derive the required features (mid_price, log_return, variance_proxy)
    features_df = derive_required_features(hour_batch)
    
    # 2) Standardization pipeline
    standardized = standardize_lob_batch(features_df)
    
    # 3) Binnarization pipeline
    binnarized = binnarize_lob_batch(standardized)
    
    # 4) Filter out overlap data (keep only current hour data for writing)
    # Get the latest hour from the binnarized data
    latest_hour = binnarized.agg({"trading_hour": "max"}).collect()[0][0]
    current_hour_only = binnarized.filter(col("trading_hour") == lit(latest_hour))
    
    cnt = current_hour_only.count()
    logger(f'process_hour_standardization: DONE, rows={cnt}')
    
    return current_hour_only.select("timestamp", "neg_bins", "pos_bins")

def write_micro_repartition(df: DataFrame, target_rows_per_part: int = 50) -> None:
    """Write data in micro-batches. Same as feature extraction script."""
    logger('write_micro_repartition: START')
    total = df.count()
    logger(f'write_micro_repartition: total={total}')

    df2 = df.coalesce(1)

    t0 = time.time()
    logger('write_micro_repartition: writing (single partition)...')
    write_append_mongo(df2)
    logger(f'write_micro_repartition: DONE in {time.time() - t0:.2f}s')

def process_hourly_standardization_batches(lob_raw: DataFrame) -> None:
    """
    Process LOB standardization and binnarization in hourly batches for memory efficiency.
    Same pattern as feature extraction script.
    """
    logger('process_hourly_standardization_batches: START')
    
    # Get all hourly ranges
    hourly_ranges = get_hourly_ranges(lob_raw)
    logger(f'process_hourly_standardization_batches: Found {len(hourly_ranges)} hourly batches')
    
    total_processed = 0
    
    for i, hour_start in enumerate(hourly_ranges):
        logger(f'process_hourly_standardization_batches: Processing batch {i+1}/{len(hourly_ranges)} - {hour_start}')
        
        # Load hour batch with overlap for continuity
        hour_batch = load_hour_batch(lob_raw, hour_start, include_overlap=True)
        
        batch_count = hour_batch.count()
        logger(f'process_hourly_standardization_batches: Batch {i+1} loaded, rows={batch_count}')
        
        if batch_count == 0:
            logger(f'process_hourly_standardization_batches: Skipping empty batch {i+1}')
            continue
        
        # Process standardization and binnarization for this hour
        hour_processed = process_hour_standardization(hour_batch)
        
        # Write processed data
        write_micro_repartition(hour_processed, target_rows_per_part=50)
        
        processed_count = hour_processed.count()
        total_processed += processed_count
        
        logger(f'process_hourly_standardization_batches: Batch {i+1} completed, processed={processed_count}, total_so_far={total_processed}')
        
        # Clean up to free memory
        hour_batch.unpersist()
        hour_processed.unpersist()
    
    logger(f'process_hourly_standardization_batches: COMPLETED all batches, total_processed={total_processed}')

# ==============================
# Main
# ==============================

if __name__ == "__main__":
    logger('MAIN: START - Hourly Standardization & Binnarization Processing')
    
    # 1) Load all LOB from landing (same as feature extraction)
    logger('MAIN: load_lob')
    lob_raw = load_lob()

    # 2) Process in hourly batches
    logger('MAIN: process_hourly_standardization_batches')
    process_hourly_standardization_batches(lob_raw)
    
    logger('MAIN: COMPLETED')