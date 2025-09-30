import sys, time
from datetime import datetime
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql.functions import col, expr, date_trunc, lit, udf, when, pow as spark_pow, row_number
from pyspark.sql.window import Window

# ==============================
# STAGE 3: STANDARDIZATION, SCALING, AND UNIFORM BINNING
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

TRANSFORMED_LOB_COLLECTION = "transformed_lob_train"
STANDARD_LOB_COLLECTION = "standard_lob_train"

spark = (
    SparkSession.builder
    .appName("Stage3_Standardization")
    .config("spark.jars", ",".join([jar_files_path + jar for jar in jar_files]))
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.mongodb.read.connection.uri", MONGO_URI)
    .config("spark.mongodb.write.connection.uri", MONGO_URI)
    .config("spark.mongodb.read.database", DB_NAME)
    .config("spark.mongodb.write.database", DB_NAME)
    .config("spark.mongodb.write.ordered", "false")
    .config("spark.mongodb.write.writeConcern.w", "1")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir("file:///C:/tmp/spark_checkpoints")

# ==============================
# Feature Identification
# ==============================

EXCLUDE_COLUMNS = [
    'timestamp', 'trading_day', 'trading_hour',
    'bids', 'asks',
    'neg_bins', 'pos_bins',
    'prices', 'volumes',
    'is_context',
    'variance_proxy',  # Used for LOB standardization, not scaled
    'mid_price',       # Used for LOB standardization, not scaled
]

EXCLUDE_PREFIXES = [
    'fwd_logret_',
]

def identify_features_to_scale(df: DataFrame) -> list:
    all_columns = df.columns
    features_to_scale = []
    
    for col_name in all_columns:
        if col_name in EXCLUDE_COLUMNS:
            continue
        if any(col_name.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            continue
        features_to_scale.append(col_name)
    
    logger(f'Identified {len(features_to_scale)} features to scale with EWMA')
    return features_to_scale

# ==============================
# IO Functions
# ==============================

def logger(msg: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[STAGE3] [{ts}] {msg}")
    sys.stdout.flush()

def load_transformed_features() -> DataFrame:
    logger(f'Loading transformed features from {TRANSFORMED_LOB_COLLECTION}')
    
    pipeline = [
        {"$sort": {"timestamp": 1}},
        {"$project": {"_id": 0}}
    ]
    
    features_df = (
        spark.read.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", TRANSFORMED_LOB_COLLECTION)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )
    
    count = features_df.count()
    logger(f'Loaded {count} transformed feature records')
    
    if 'is_context' in features_df.columns:
        context_count = features_df.filter(col('is_context') == True).count()
        non_context_count = features_df.filter(col('is_context') == False).count()
        logger(f'Context breakdown: {context_count} context + {non_context_count} non-context rows')
    
    return features_df

def get_processable_hours(df: DataFrame, required_past_hours: int = 3) -> list:
    logger('Analyzing processable hours...')
    
    all_hours = (df.withColumn("trading_hour", date_trunc("hour", col("timestamp")))
                   .select("trading_hour")
                   .distinct()
                   .orderBy("trading_hour")
                   .collect())
    
    available_hours = [row.trading_hour for row in all_hours]
    logger(f'Available hours: {len(available_hours)} from {available_hours[0]} to {available_hours[-1]}')
    
    processable_hours = []
    for i, hour in enumerate(available_hours):
        past_hours_available = i
        if past_hours_available >= required_past_hours:
            processable_hours.append(hour)
    
    logger(f'Processable hours (with {required_past_hours}h lookback): {len(processable_hours)}')
    if processable_hours:
        logger(f'Range: {processable_hours[0]} to {processable_hours[-1]}')
    
    return processable_hours

def load_hour_batch(df: DataFrame, target_hour, required_past_hours: int = 3):
    past_start = expr(f"'{target_hour}' - INTERVAL {required_past_hours} HOURS")
    future_end = expr(f"'{target_hour}' + INTERVAL 1 HOURS")
    
    filter_condition = (
        (col("timestamp") >= past_start) &
        (col("timestamp") < future_end)
    )
    
    logger(f'Loading batch for target hour {target_hour} with {required_past_hours}h lookback')
    
    batch = df.filter(filter_condition)
    
    if 'is_context' in batch.columns:
        context_in_batch = batch.filter(col('is_context') == True).count()
        target_in_batch = batch.filter(col('is_context') == False).count()
        logger(f'Batch breakdown: {context_in_batch} context + {target_in_batch} target rows')
    
    return batch

def write_standard_to_mongo(df: DataFrame) -> None:
    (df.write.format("mongodb")
       .option("database", DB_NAME)
       .option("collection", STANDARD_LOB_COLLECTION)
       .option("ordered", "false")
       .mode("append")
       .save())

# ==============================
# LOB Standardization Functions
# ==============================

def standardize_prices(df: DataFrame) -> DataFrame:
    eps = 1e-8
    min_denom = 1e-6
    
    df = df.withColumn(
        "_safe_denom",
        expr(f"GREATEST(sqrt(variance_proxy) * mid_price, {min_denom})")
    )
    
    df = df.withColumn(
        "bids", 
        expr(f"transform(bids, x -> array((x[0] - mid_price) / (_safe_denom + {eps}), x[1]))")
    ).withColumn(
        "asks", 
        expr(f"transform(asks, x -> array((x[0] - mid_price) / (_safe_denom + {eps}), x[1]))")
    )
    
    return df.drop("_safe_denom")

def normalize_volumes(df: DataFrame) -> DataFrame:
    df = df.withColumn(
        "total_volume",
        expr("aggregate(bids, 0D, (acc, x) -> acc + x[1]) + aggregate(asks, 0D, (acc, x) -> acc + x[1])")
    )

    df = df.withColumn(
        "bids", 
        expr("transform(bids, x -> array(x[0], CASE WHEN total_volume = 0 THEN 0D ELSE x[1] / total_volume END))")
    ).withColumn(
        "asks", 
        expr("transform(asks, x -> array(x[0], CASE WHEN total_volume = 0 THEN 0D ELSE x[1] / total_volume END))")
    )

    return df.drop("total_volume")

def flatten_bids_and_asks(df: DataFrame) -> DataFrame:
    df = df.withColumn("pairs", expr("concat(bids, asks)"))
    df = df.withColumn("prices", expr("transform(pairs, x -> x[0])"))
    df = df.withColumn("volumes", expr("transform(pairs, x -> x[1])"))
    
    return df.drop("bids", "asks", "pairs")

# ==============================
# Uniform Quantization Functions
# ==============================

def uniform_quantize_and_aggregate(df: DataFrame, B: int, price_range_sigma: float) -> DataFrame:
    logger(f'Performing per-snapshot uniform quantization with B={B} bins...')
    
    B_broadcast = spark.sparkContext.broadcast(B)
    
    @udf(ArrayType(ArrayType(DoubleType())))
    def bin_volumes_per_snapshot(prices_arr, volumes_arr):
        B_local = B_broadcast.value
        B_half = B_local // 2
        
        empty_result = [[0.0] * B_half, [0.0] * B_half]
        
        if not prices_arr or not volumes_arr:
            return empty_result
        
        prices = np.array(prices_arr, dtype=np.float64)
        volumes = np.array(volumes_arr, dtype=np.float64)
        
        n = min(len(prices), len(volumes))
        if n == 0:
            return empty_result
        
        prices = prices[:n]
        volumes = volumes[:n]
        
        valid_mask = np.isfinite(prices) & np.isfinite(volumes) & (volumes > 0)
        prices = prices[valid_mask]
        volumes = volumes[valid_mask]
        
        if len(prices) == 0:
            return empty_result
        
        min_price = np.min(prices)
        max_price = np.max(prices)
        
        if max_price - min_price < 1e-10:
            bins = np.zeros(B_local, dtype=np.float64)
            bins[B_local // 2] = np.sum(volumes)
            neg_bins = bins[:B_half].tolist()
            pos_bins = bins[B_half:].tolist()
            return [neg_bins, pos_bins]
        
        edges = np.linspace(min_price, max_price, B_local + 1)
        
        idx = np.digitize(prices, edges, right=False) - 1
        idx = np.clip(idx, 0, B_local - 1)
        
        bins = np.zeros(B_local, dtype=np.float64)
        np.add.at(bins, idx, volumes)
        
        zero_idx = np.searchsorted(edges, 0.0)
        zero_bin_idx = zero_idx - 1
        zero_bin_idx = np.clip(zero_bin_idx, 0, B_local)
        
        neg_bins_raw = bins[:zero_bin_idx]
        pos_bins_raw = bins[zero_bin_idx:]
        
        if len(neg_bins_raw) < B_half:
            neg_bins = np.concatenate([
                np.zeros(B_half - len(neg_bins_raw)),
                neg_bins_raw
            ])
        else:
            neg_bins = neg_bins_raw[-B_half:]
        
        if len(pos_bins_raw) < B_half:
            pos_bins = np.concatenate([
                pos_bins_raw,
                np.zeros(B_half - len(pos_bins_raw))
            ])
        else:
            pos_bins = pos_bins_raw[:B_half]
        
        return [neg_bins.tolist(), pos_bins.tolist()]
    
    df = df.withColumn("binned", bin_volumes_per_snapshot(col("prices"), col("volumes")))
    df = df.withColumn("neg_bins", expr("binned[0]"))
    df = df.withColumn("pos_bins", expr("binned[1]"))
    
    logger(f'Created {B//2} negative bins and {B//2} positive bins per snapshot')
    
    return df.drop("binned")

# ==============================
# EWMA Scaling for Features
# ==============================

def scale_feature_with_ewma(df: DataFrame, feature_name: str, half_life: int = 20, clip_std: float = 3.0) -> DataFrame:
    alpha = 1.0 - spark_pow(lit(2.0), lit(-1.0) / lit(float(half_life)))
    beta = lit(1.0) - alpha
    
    w = Window.orderBy("timestamp")
    wcum = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    rn = row_number().over(w) - lit(1)
    
    feature_col = col(feature_name)
    
    df = df.withColumn("_z_mean", when(feature_col.isNotNull(), feature_col * spark_pow(beta, -rn)).otherwise(lit(None)))
    df = df.withColumn("_cum_mean", expr("sum(_z_mean)").over(wcum))
    df = df.withColumn("_ewma_mean", alpha * spark_pow(beta, rn) * col("_cum_mean"))
    
    df = df.withColumn("_centered", feature_col - col("_ewma_mean"))
    
    df = df.withColumn("_centered_sq", col("_centered") * col("_centered"))
    df = df.withColumn("_z_var", when(col("_centered_sq").isNotNull(), col("_centered_sq") * spark_pow(beta, -rn)).otherwise(lit(None)))
    df = df.withColumn("_cum_var", expr("sum(_z_var)").over(wcum))
    df = df.withColumn("_ewma_var", alpha * spark_pow(beta, rn) * col("_cum_var"))
    df = df.withColumn("_ewma_std", expr("sqrt(_ewma_var + 1e-12)"))
    
    df = df.withColumn("_scaled", col("_centered") / col("_ewma_std"))
    
    df = df.withColumn(
        feature_name,
        when(col("_scaled") > clip_std, lit(clip_std))
        .when(col("_scaled") < -clip_std, lit(-clip_std))
        .otherwise(col("_scaled"))
    )
    
    df = df.drop("_z_mean", "_cum_mean", "_ewma_mean", "_centered", "_centered_sq", "_z_var", "_cum_var", "_ewma_var", "_ewma_std", "_scaled")
    
    return df

def scale_all_features_with_ewma(df: DataFrame, features_to_scale: list, half_life: int = 20, clip_std: float = 3.0) -> DataFrame:
    logger(f'Scaling {len(features_to_scale)} features with EWMA (half_life={half_life}, clip={clip_std}σ)...')
    
    result_df = df
    checkpoint_interval = 5
    
    for i, feature_name in enumerate(features_to_scale):
        if feature_name not in df.columns:
            logger(f'Warning: {feature_name} not in dataframe, skipping')
            continue
        
        logger(f'  [{i+1}/{len(features_to_scale)}] Scaling {feature_name}...')
        result_df = scale_feature_with_ewma(result_df, feature_name, half_life, clip_std)
        
        if (i + 1) % checkpoint_interval == 0:
            logger(f'  Checkpointing after {i+1} features...')
            result_df = result_df.localCheckpoint(eager=False)
    
    logger('Final checkpoint of scaled features...')
    result_df = result_df.localCheckpoint(eager=True)
    
    logger('Feature scaling complete')
    return result_df

# ==============================
# Main Processing Pipeline
# ==============================

def standardize_scale_and_bin_hour(hour_features: DataFrame, target_hour, B: int = 1000, price_range_sigma: float = 5.0, half_life: int = 20, clip_std: float = 3.0) -> DataFrame:
    logger(f'Processing standardization for target hour: {target_hour}')
    
    if "bids" not in hour_features.columns or "asks" not in hour_features.columns:
        logger(f'ERROR - Missing LOB data for {target_hour}')
        return hour_features.limit(0)
    
    df = hour_features
    
    logger('Standardizing LOB prices...')
    df = standardize_prices(df)
    
    logger('Normalizing volumes...')
    df = normalize_volumes(df)
    
    logger('Flattening bids and asks...')
    df = flatten_bids_and_asks(df)
    
    logger('Performing uniform quantization...')
    df = uniform_quantize_and_aggregate(df, B=B, price_range_sigma=price_range_sigma)
    
    features_to_scale = identify_features_to_scale(df)
    logger('Scaling derived features with EWMA...')
    df = scale_all_features_with_ewma(df, features_to_scale, half_life, clip_std)
    
    logger(f'Filtering to target hour: {target_hour}')
    df = df.filter(col("trading_hour") == lit(target_hour))
    
    if 'is_context' in df.columns:
        df = df.filter(col("is_context") == False)
    
    keep_cols = ['timestamp', 'trading_day', 'trading_hour']
    keep_cols += ['neg_bins', 'pos_bins']
    keep_cols += [c for c in df.columns if c.startswith('fwd_logret_')]
    keep_cols += features_to_scale
    
    keep_cols = [c for c in keep_cols if c in df.columns and c != 'is_context']
    
    df = df.select(*keep_cols)
    df = df.repartition(1).sortWithinPartitions("timestamp")
    
    cnt = df.count()
    logger(f'Standardization complete - target hour {target_hour}, rows={cnt}')
    
    return df

def process_stage3_hourly_batches(features_df: DataFrame, B: int = 1000, price_range_sigma: float = 5.0, half_life: int = 20, clip_std: float = 3.0, required_past_hours: int = 3) -> None:
    logger('STAGE 3 STANDARDIZATION, SCALING & BINNING: START')
    logger(f'Parameters: B={B}, price_range=±{price_range_sigma}σ, half_life={half_life}, clip_std={clip_std}σ, lookback={required_past_hours}h')
    
    processable_hours = get_processable_hours(features_df, required_past_hours=required_past_hours)
    
    if not processable_hours:
        logger('ERROR - No processable hours found!')
        return
    
    logger(f'Processing {len(processable_hours)} hours')
    logger(f'Range: {processable_hours[0]} to {processable_hours[-1]}')
    
    total_processed = 0
    total_time = 0
    
    for i, target_hour in enumerate(processable_hours):
        batch_start_time = time.time()
        logger(f'Processing {i+1}/{len(processable_hours)} - TARGET HOUR: {target_hour}')
        
        hour_batch = load_hour_batch(features_df, target_hour, required_past_hours=required_past_hours)
        batch_count = hour_batch.count()
        logger(f'Loaded {batch_count} rows')
        
        if batch_count > 0:
            standard_df = standardize_scale_and_bin_hour(hour_batch, target_hour, B=B, price_range_sigma=price_range_sigma, half_life=half_life, clip_std=clip_std)
            
            logger('Writing to MongoDB...')
            standard_df_coalesced = standard_df.coalesce(1)
            write_standard_to_mongo(standard_df_coalesced)
            
            processed_count = standard_df.count()
            total_processed += processed_count
            logger(f'Wrote {processed_count} rows')
            
            standard_df.unpersist()
        else:
            logger(f'Skipping empty batch for {target_hour}')
        
        batch_duration = time.time() - batch_start_time
        total_time += batch_duration
        avg_time = total_time / (i + 1)
        eta = avg_time * (len(processable_hours) - i - 1)
        
        logger(f'Batch completed in {batch_duration:.2f}s, ETA: {eta:.2f}s')
        
        hour_batch.unpersist()
    
    logger(f'STAGE 3 COMPLETED - Total: {total_processed} rows in {total_time:.2f}s')

# ==============================
# Main Execution
# ==============================

if __name__ == "__main__":
    start_time = time.time()
    
    logger('STAGE 3: Standardization, EWMA Scaling, and Uniform Binning Pipeline')
    logger(f'Input: {TRANSFORMED_LOB_COLLECTION}')
    logger(f'Output: {STANDARD_LOB_COLLECTION}')
    
    logger('Loading transformed features...')
    transformed_features = load_transformed_features()
    
    logger('Processing standardization, binning, and scaling...')
    process_stage3_hourly_batches(
        transformed_features, 
        B=1000,
        price_range_sigma=5.0,
        half_life=20,
        clip_std=3.0,
        required_past_hours=3
    )
    
    total_time = time.time() - start_time
    logger(f'STAGE 3 completed in {total_time:.2f} seconds')
    
    spark.stop()