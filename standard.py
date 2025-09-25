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
    date_trunc, approx_count_distinct, udf, collect_list, lead
)

# ==============================
# STAGE 3: STANDARDIZATION AND BINNARIZATION
# ==============================

jar_files_path = "file:///C:/Users/llucp/spark_jars/"

jar_files = [
    "mongo-spark-connector_2.12-10.1.1.jar",
    "mongodb-driver-core-4.10.1.jar",
    "mongodb-driver-sync-4.10.1.jar",
    "bson-4.10.1.jar"
]

# Updated architecture - Stage 3 reads from features, writes to standardized
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "tfg"

# Input: Stage 2 output
FEATURES_LOB_COLLECTION = "features_lob"

# Output: Stage 3 standardized
STANDARD_LOB_COLLECTION = "standard_lob"

spark = (
    SparkSession.builder
    .appName("Stage3_Standardization")
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
# Stage 3 IO Functions
# ==============================

def logger(msg: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[STAGE3] [{ts}] {msg}")
    sys.stdout.flush()

def load_features_lob(limit: int = None) -> DataFrame:
    """
    STAGE 3: Loads feature-engineered LOB data from Stage 2 output.
    """
    logger(f'Loading features LOB data from {FEATURES_LOB_COLLECTION}')
    
    pipeline = [
        {"$sort": {"timestamp": 1}},  # Ensure chronological order
        {"$project": {"_id": 0}}  # Project all fields except _id
    ]
    
    # Add limit if specified (for testing)
    if limit:
        pipeline.insert(1, {"$limit": limit})
    
    features_lob = (
        spark.read.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", FEATURES_LOB_COLLECTION)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )
    
    count = features_lob.count()
    logger(f'Loaded {count} feature records')
    return features_lob

def get_processable_hours(df: DataFrame, required_past_hours: int = 0, required_future_hours: int = 0) -> list:
    """
    Get all hours that can be processed for standardization.
    Since features are already calculated, we don't need context windows for Stage 3.
    """
    logger('get_processable_hours: START')
    
    # Get all available hours in chronological order
    all_hours = (df.withColumn("trading_hour", date_trunc("hour", col("timestamp")))
                   .select("trading_hour")
                   .distinct()
                   .orderBy("trading_hour")
                   .collect())
    
    available_hours = [row.trading_hour for row in all_hours]
    logger(f'Available hours: {len(available_hours)} hours from {available_hours[0]} to {available_hours[-1]}')
    
    # For Stage 3, all hours with features are processable
    processable_hours = available_hours
    
    logger(f'Processable hours: {len(processable_hours)} hours')
    if processable_hours:
        logger(f'Range: {processable_hours[0]} to {processable_hours[-1]}')
    
    return processable_hours

def load_hour_batch(df: DataFrame, target_hour):
    """
    STAGE 3: Load data batch for processing a specific target hour.
    No context needed since features are already calculated.
    """
    filter_condition = (
        col("trading_hour") == lit(target_hour)
    )
    
    logger(f'Loading batch for target hour {target_hour}')
    
    return df.filter(filter_condition)

def write_standard_to_mongo(df: DataFrame) -> None:
    """
    STAGE 3: Write standardized data to standard collection.
    """
    (df.write.format("mongodb")
       .option("database", DB_NAME)
       .option("collection", STANDARD_LOB_COLLECTION)
       .option("ordered", "false")
       .mode("append")
       .save())

# ==============================
# LOB Standardization Functions (from your working script)
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

# ==============================
# LOB Binnarization Functions (from your working script)
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

# ==============================
# Stage 3 Main Processing Pipeline
# ==============================

def standardize_and_binnarize_hour(hour_features: DataFrame, target_hour) -> DataFrame:
    """
    STAGE 3: Standardize and binnarize features for a single hour.
    
    Args:
        hour_features: DataFrame with feature-engineered data for target hour
        target_hour: The specific hour being processed
    """
    logger(f'Processing standardization for target hour: {target_hour}')
    
    # Verify we have the expected data structure
    if "bids" not in hour_features.columns or "asks" not in hour_features.columns:
        logger(f'ERROR - Missing LOB data (bids/asks) for {target_hour}')
        return hour_features.limit(0)
    
    # Step 1: Standardization pipeline
    logger('Standardizing LOB prices...')
    df = standardize_prices(hour_features)
    
    logger('Normalizing volumes...')
    df = normalize_volumes(df)
    
    logger('Flattening bids and asks...')
    df = flatten_bids_and_asks(df)
    
    logger('Calculating gamma scaling factor...')
    df = calculate_gamma_scaling_factor_sql(df, L=10.0, target=0.95)
    
    logger('Applying gamma scaling...')
    df = gamma_scale_std_prices(df)
    
    # Step 2: Binnarization pipeline
    logger('Quantizing LOB data...')
    best_delta, edges, best_snr = quantize_lob_batch(
        df, L=10.0, B=1000, w_factor=0.1, clip_max_percentile=0.999
    )
    
    logger('Aggregating volume per bin...')
    df = aggregate_volume_per_bin_sql(df, edges)
    
    logger('Splitting bins...')
    df = split_bins_sql(df, edges)
    
    # Step 3: Combine results
    # Keep features but replace LOB data with binnarized version
    feature_cols = [c for c in hour_features.columns if c not in ["bids", "asks"]]
    features_only = hour_features.select(*feature_cols)
    bins_only = df.select("timestamp", "neg_bins", "pos_bins")
    
    final_df = features_only.join(bins_only, on="timestamp", how="inner")
    final_df = final_df.repartition(1).sortWithinPartitions("timestamp")
    
    cnt = final_df.count()
    logger(f'Standardization complete - target hour {target_hour}, rows={cnt}')
    
    return final_df

# ==============================
# Stage 3 Batch Processing
# ==============================

def write_micro_repartition(df: DataFrame, target_rows_per_part: int = 50) -> None:
    logger('Writing standardized data to MongoDB...')
    total = df.count()
    logger(f'Total rows to write: {total}')

    # Coalesce to single partition for small batches
    df2 = df.coalesce(1)

    t0 = time.time()
    write_standard_to_mongo(df2)
    logger(f'Write completed in {time.time() - t0:.2f}s')

def process_stage3_hourly_batches(features_lob: DataFrame, limit_hours: int = None) -> None:
    """
    STAGE 3: Process feature-engineered data into standardized/binnarized data.
    """
    logger('STAGE 3 STANDARDIZATION & BINNARIZATION: START')
    
    # Get all processable hours (all hours with features)
    processable_hours = get_processable_hours(features_lob)
    
    if not processable_hours:
        logger('ERROR - No processable hours found!')
        return
    
    # Limit for testing if specified
    if limit_hours:
        processable_hours = processable_hours[:limit_hours]
        logger(f'Limited processing to first {limit_hours} hours for testing')
    
    logger(f'Processing {len(processable_hours)} hours')
    logger(f'First processable hour: {processable_hours[0]}')
    logger(f'Last processable hour: {processable_hours[-1]}')
    
    total_processed = 0
    total_time = 0
    
    for i, target_hour in enumerate(processable_hours):
        batch_start_time = time.time()
        logger(f'Processing {i+1}/{len(processable_hours)} - TARGET HOUR: {target_hour}')
        
        # Load batch for target hour
        hour_batch = load_hour_batch(features_lob, target_hour)
        
        batch_count = hour_batch.count()
        logger(f'Loaded batch for target hour {target_hour} - {batch_count} rows')
        
        if batch_count > 0:
            # Process standardization and binnarization
            standard_df = standardize_and_binnarize_hour(hour_batch, target_hour)
            
            # Verify correct hour
            processed_hours = (standard_df.select("trading_hour").distinct().collect())
            actual_hours = [row.trading_hour for row in processed_hours]
            
            if len(actual_hours) == 1 and actual_hours[0] == target_hour:
                logger(f'✓ Correctly processed target hour: {target_hour}')
            else:
                logger(f'✗ ERROR - Expected {target_hour}, got {actual_hours}')
            
            # Write to standard collection
            write_micro_repartition(standard_df, target_rows_per_part=50)
            
            processed_count = standard_df.count()
            total_processed += processed_count
            
            standard_df.unpersist()
        else:
            logger(f'Skipping empty batch for {target_hour}')
        
        # Timing
        batch_duration = time.time() - batch_start_time
        total_time += batch_duration
        avg_time_per_batch = total_time / (i + 1)
        estimated_remaining = avg_time_per_batch * (len(processable_hours) - i - 1)
        
        logger(f'Batch completed in {batch_duration:.2f}s')
        logger(f'Progress: {i+1}/{len(processable_hours)}, ETA: {estimated_remaining:.2f}s')
        
        hour_batch.unpersist()
    
    logger(f'STAGE 3 COMPLETED - Total processed: {total_processed} rows in {total_time:.2f}s')

# ==============================
# Main Stage 3 Execution
# ==============================

if __name__ == "__main__":
    start_time = time.time()
    
    logger('STAGE 3: Standardization and Binnarization Pipeline')
    logger(f'Input: {FEATURES_LOB_COLLECTION}')
    logger(f'Output: {STANDARD_LOB_COLLECTION}')
    
    # Load features data from Stage 2
    logger('Loading features LOB data from Stage 2...')
    features_lob = load_features_lob(limit=1000)  # Limit for testing, remove for full processing
    
    # Process standardization in hourly batches
    logger('Processing standardization and binnarization in hourly batches...')
    process_stage3_hourly_batches(features_lob, limit_hours=3)  # Process first 3 hours for testing
    
    total_time = time.time() - start_time
    logger(f'STAGE 3 completed in {total_time:.2f} seconds')
    logger('Pipeline complete: raw_lob -> features_lob -> standard_lob')
    
    spark.stop()