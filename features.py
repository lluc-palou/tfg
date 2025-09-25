import math
import sys, time
from datetime import datetime
import numpy as np
from pyspark import StorageLevel
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import LongType
from pyspark.sql.functions import (
    col, expr, log, lag, lead, when, to_date, pow, row_number, lit, sum as spark_sum, 
    date_trunc, approx_count_distinct
)

# ==============================
# STAGE 2: FEATURE ENGINEERING
# ==============================

jar_files_path = "file:///C:/Users/llucp/spark_jars/"

jar_files = [
    "mongo-spark-connector_2.12-10.1.1.jar",
    "mongodb-driver-core-4.10.1.jar",
    "mongodb-driver-sync-4.10.1.jar",
    "bson-4.10.1.jar"
]

# Updated architecture - Stage 2 reads from raw, writes to features
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "tfg"

# Input: Stage 1 output
RAW_LOB_COLLECTION = "raw_lob"

# Output: Stage 2 features
FEATURES_LOB_COLLECTION = "features_lob"

spark = (
    SparkSession.builder
    .appName("Stage2_FeatureEngineering")
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
# Stage 2 IO Functions
# ==============================

def logger(msg: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[STAGE2] [{ts}] {msg}")
    sys.stdout.flush()

def load_raw_lob(limit: int = None) -> DataFrame:
    """
    STAGE 2: Loads raw LOB data from Stage 1 output.
    """
    logger(f'Loading raw LOB data from {RAW_LOB_COLLECTION}')
    
    pipeline = [
        {"$sort": {"timestamp": 1}},  # Ensure chronological order
        {"$project": {"timestamp": 1, "bids": 1, "asks": 1}}
    ]
    
    # Add limit if specified (for testing)
    if limit:
        pipeline.insert(1, {"$limit": limit})
    
    raw_lob = (
        spark.read.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", RAW_LOB_COLLECTION)
        .option("aggregation.pipeline", str(pipeline).replace("'", '"'))
        .load()
    )
    
    count = raw_lob.count()
    logger(f'Loaded {count} raw LOB records')
    return raw_lob

def get_processable_hours(df: DataFrame, required_past_hours: int = 3, required_future_hours: int = 3) -> list:
    """
    Get all hours that can be fully processed (have sufficient past and future context).
    Returns list of processable hours in chronological order.
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
    
    # Filter hours that have sufficient context
    processable_hours = []
    for i, hour in enumerate(available_hours):
        past_hours_available = i  # Hours before current
        future_hours_available = len(available_hours) - i - 1  # Hours after current
        
        if past_hours_available >= required_past_hours and future_hours_available >= required_future_hours:
            processable_hours.append(hour)
    
    logger(f'Processable hours: {len(processable_hours)} hours')
    if processable_hours:
        logger(f'Range: {processable_hours[0]} to {processable_hours[-1]}')
    
    return processable_hours

def load_hour_batch(df: DataFrame, target_hour, required_past_hours: int = 3, required_future_hours: int = 3):
    """
    STAGE 2: Load data batch for processing a specific target hour.
    Includes required past hours for features and future hours for targets.
    """
    # Calculate the actual time window needed
    past_start = expr(f"'{target_hour}' - INTERVAL {required_past_hours} HOURS")
    future_end = expr(f"'{target_hour}' + INTERVAL {required_future_hours + 1} HOURS")
    
    filter_condition = (
        (col("timestamp") >= past_start) &
        (col("timestamp") < future_end)
    )
    
    logger(f'Loading batch for target hour {target_hour}')
    logger(f'Window: {target_hour} - {required_past_hours}h to {target_hour} + {required_future_hours + 1}h')
    
    return df.filter(filter_condition)

def write_features_to_mongo(df: DataFrame) -> None:
    """
    STAGE 2: Write feature-engineered data to features collection.
    """
    (df.write.format("mongodb")
       .option("database", DB_NAME)
       .option("collection", FEATURES_LOB_COLLECTION)
       .option("ordered", "false")
       .mode("append")
       .save())

# ==============================
# Target Derivation (Forward Returns)
# ==============================

def calculate_forward_log_returns_midprice(df: DataFrame, lag: int, N: int) -> DataFrame:
    """
    Calculates forward log returns of mid_price over N periods, accounting for a decision lag.
    """
    w = Window.orderBy("timestamp")
    base = lead(col("mid_price"), lag).over(w)
    future = lead(col("mid_price"), lag + N).over(w)

    return df.withColumn(f"fwd_logret_{N}", log(future) - log(base))

def calculate_past_log_returns_midprice(df: DataFrame, N: int) -> DataFrame:
    """
    Calculates historical log returns of mid_price over N snapshots.
    """
    w = Window.orderBy("timestamp")
    past = lag(col("mid_price"), N).over(w)
    return df.withColumn(f"past_logret_{N}", log(col("mid_price")) - log(past))

def derive_midprice_targets(df: DataFrame) -> DataFrame:
    """
    Derives targets as forward log-returns of mid_price over multiple horizons.
    """
    logger('derive_midprice_targets: START')
    H = [2, 3, 4, 5, 10, 20, 40, 60, 120, 240]
    lag = 1

    result = df
    for N in H:
        logger(f'calculating fwd_logret_{N}...')
        result = calculate_forward_log_returns_midprice(result, lag=lag, N=N)

    logger('derive_midprice_targets: DONE')
    return result

def derive_midprice_past_returns(df: DataFrame) -> DataFrame:
    """
    Derives historical log returns of mid_price over multiple past horizons.
    """
    logger('derive_midprice_past_returns: START')
    H = [1, 2, 3, 4, 5, 10, 20, 40, 60, 120, 240]

    result = df
    for N in H:
        logger(f'calculating past_logret_{N}...')
        result = calculate_past_log_returns_midprice(result, N=N)

    logger('derive_midprice_past_returns: DONE')
    return result

# ==============================
# Feature Derivation Functions (same as before)
# ==============================

def calculate_mid_prices(df: DataFrame) -> DataFrame:
    best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
    best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")
    return df.withColumn("mid_price", (best_bid + best_ask) / 2.0)

def calculate_log_returns(df: DataFrame) -> DataFrame:
    """
    BATCH-LOCAL log-returns ordered by timestamp within the batch.
    """
    w = Window.orderBy("timestamp")
    return df.withColumn("log_return", log(col("mid_price")) - log(lag(col("mid_price"), 1).over(w)))

def estimate_variance(df: DataFrame, half_life: int) -> DataFrame:
    """
    BATCH-LOCAL SQL-only EWMA of squared log-returns.
    """
    alpha = 1.0 - pow(lit(2.0), lit(-1.0) / lit(float(half_life)))
    beta  = (lit(1.0) - alpha)

    w = Window.orderBy("timestamp")
    rn = row_number().over(w) - lit(1)  # 0-based index

    df = df.withColumn("r2", col("log_return") * col("log_return"))
    df = df.withColumn("z_i", when(col("r2").isNotNull(), col("r2") * pow(beta, -rn)).otherwise(lit(None)))

    wcum = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df = df.withColumn("cum_z", spark_sum("z_i").over(wcum))

    df = df.withColumn("variance_proxy", alpha * pow(beta, rn) * col("cum_z"))
    df = df.drop("r2", "z_i", "cum_z")
    return df

def calculate_volatility(df: DataFrame) -> DataFrame:
    return df.withColumn("volatility", expr("sqrt(variance_proxy)"))

def calculate_spread(df: DataFrame) -> DataFrame:
    best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
    best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")
    return df.withColumn("spread", best_ask - best_bid)

def calculate_microprice(df: DataFrame) -> DataFrame:
    best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
    best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")
    df = df.withColumn("best_bid", best_bid).withColumn("best_ask", best_ask)

    best_bid_vol = expr("aggregate(filter(bids, x -> x[0] = best_bid), 0D, (acc,x) -> acc + x[1])")
    best_ask_vol = expr("aggregate(filter(asks, x -> x[0] = best_ask), 0D, (acc,x) -> acc + x[1])")
    df = df.withColumn("best_bid_vol", best_bid_vol).withColumn("best_ask_vol", best_ask_vol)

    micro_expr = when(
        (col("best_bid_vol") + col("best_ask_vol")) == 0.0,
        (col("best_bid") + col("best_ask")) / 2.0
    ).otherwise(
        (col("best_ask") * col("best_bid_vol") + col("best_bid") * col("best_ask_vol")) /
        (col("best_bid_vol") + col("best_ask_vol"))
    )

    df = df.withColumn("microprice", micro_expr)
    return df.drop("best_bid", "best_ask", "best_bid_vol", "best_ask_vol")

def calculate_depth_imbalance(df: DataFrame, k: int) -> DataFrame:
    K = int(k)
    b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
    a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
    b_desc = f"reverse(array_sort({b_struct}))"
    a_asc  = f"array_sort({a_struct})"
    b_topk = f"slice({b_desc}, 1, {K})" if K > 0 else b_desc
    a_topk = f"slice({a_asc},  1, {K})" if K > 0 else a_asc
    vb = f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y) -> acc + y)"
    va = f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y) -> acc + y)"
    denom = f"(({vb}) + ({va}))"
    imb = f"CASE WHEN ({denom})=0D THEN 0D ELSE (({vb}) - ({va})) / ({denom}) END"
    suffix = f"top_{K}" if K > 0 else "all"
    return df.withColumn(f"depth_imbalance_{suffix}", expr(imb))

def calculate_market_depth(df: DataFrame, k: int) -> DataFrame:
    K = int(k)
    b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
    a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
    b_desc = f"reverse(array_sort({b_struct}))"
    a_asc  = f"array_sort({a_struct})"
    b_topk = f"slice({b_desc}, 1, {K})" if K > 0 else b_desc
    a_topk = f"slice({a_asc},  1, {K})" if K > 0 else a_asc
    depth_sum = (
        f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y)->acc+y) + "
        f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y)->acc+y)"
    )
    suffix = f"top_{K}" if K > 0 else "all"
    return df.withColumn(f"depth_{suffix}", expr(depth_sum))

def calculate_liquidity_concentration(df: DataFrame, k: int) -> DataFrame:
    K = int(k)
    b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
    a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
    b_desc = f"reverse(array_sort({b_struct}))"
    a_asc  = f"array_sort({a_struct})"
    b_topk = f"slice({b_desc}, 1, {K})" if K > 0 else b_desc
    a_topk = f"slice({a_asc},  1, {K})" if K > 0 else a_asc
    denom = (
        f"aggregate(transform({b_desc}, x -> x.v), 0D, (acc,y)->acc+y) + "
        f"aggregate(transform({a_asc}, x -> x.v), 0D, (acc,y)->acc+y)"
    )
    num = (
        f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y)->acc+y) + "
        f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y)->acc+y)"
    )
    lc = f"CASE WHEN ({denom})=0D THEN 0D ELSE ({num})/({denom}) END"
    suffix = f"top_{K}" if K > 0 else "all"
    return df.withColumn(f"liquidity_concentration_{suffix}", expr(lc))

def calculate_price_impact_proxy(df: DataFrame, k: int, eps: float = 1e-12) -> DataFrame:
    K = int(k)
    b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
    a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
    b_desc = f"reverse(array_sort({b_struct}))"   # best at index 1
    a_asc  = f"array_sort({a_struct})"
    b_eff = f"CASE WHEN {K} > 0 THEN least({K}, size({b_desc})) ELSE size({b_desc}) END"
    a_eff = f"CASE WHEN {K} > 0 THEN least({K}, size({a_asc}))  ELSE size({a_asc})  END"
    PbK = f"element_at({b_desc}, {b_eff}).p"
    PaK = f"element_at({a_asc},  {a_eff}).p"
    b_topk = f"slice({b_desc}, 1, {b_eff})"
    a_topk = f"slice({a_asc},  1, {a_eff})"
    depth_sum = (
        f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y)->acc+y) + "
        f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y)->acc+y)"
    )
    lam = f"CASE WHEN ({depth_sum})<=0D THEN 0D ELSE (({PaK}) - ({PbK})) / ({depth_sum} + {eps}) END"
    suffix = f"top_{K}" if K > 0 else "all"
    return df.withColumn(f"price_impact_proxy_{suffix}", expr(lam))

def calculate_liquidity_spread(df: DataFrame, k: int, eps: float = 1e-12) -> DataFrame:
    K = int(k)
    b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
    a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
    b_desc = f"reverse(array_sort({b_struct}))"
    a_asc  = f"array_sort({a_struct})"
    kmin_all = f"least(size({b_desc}), size({a_asc}))"
    k_eff = f"CASE WHEN {K} > 0 THEN least({K}, {kmin_all}) ELSE {kmin_all} END"
    b_top = f"slice({b_desc}, 1, {k_eff})"
    a_top = f"slice({a_asc},  1, {k_eff})"
    Pa = f"transform({a_top}, x -> x.p)"
    Va = f"transform({a_top}, x -> x.v)"
    Pb = f"transform({b_top}, x -> x.p)"
    Vb = f"transform({b_top}, x -> x.v)"
    w    = f"zip_with({Va}, {Vb}, (va, vb) -> va + vb)"
    diff = f"zip_with({Pa}, {Pb}, (pa, pb) -> pa - pb)"
    num_arr = f"zip_with({diff}, {w}, (d, ww) -> d * ww)"
    denom = f"aggregate({w}, 0D, (acc, y) -> acc + y)"
    num   = f"aggregate({num_arr}, 0D, (acc, y) -> acc + y)"
    lspread = f"CASE WHEN ({denom})<=0D THEN 0D ELSE ({num}) / ({denom} + {eps}) END"
    suffix = f"top_{K}" if K > 0 else "all"
    return df.withColumn(f"liquidity_spread_{suffix}", expr(lspread))

# ==============================
# Stage 2 Main Feature Engineering Pipeline
# ==============================

def derive_hour_features(hour_batch: DataFrame, target_hour) -> DataFrame:
    """
    STAGE 2: Process features for a single hour batch, ensuring only the target hour is returned.
    Includes both features AND targets (forward returns).
    """
    logger(f'Processing features for target hour: {target_hour}')
    
    # Verify target hour exists in batch
    batch_hours = (hour_batch.withColumn("trading_hour", date_trunc("hour", col("timestamp")))
                            .select("trading_hour")
                            .distinct()
                            .collect())
    available_hours = [row.trading_hour for row in batch_hours]
    available_hours.sort()
    
    logger(f'Batch contains hours: {available_hours}')
    
    if target_hour not in available_hours:
        logger(f'ERROR - Target hour {target_hour} not found in batch!')
        return hour_batch.limit(0)
    
    # Prepare base data (keep all hours for feature/target calculation)
    base = (
        hour_batch.select("timestamp","bids","asks")
                  .withColumn("trading_day", to_date(col("timestamp")))
                  .withColumn("trading_hour", date_trunc("hour", col("timestamp")))
                  .repartition(4)
                  .sortWithinPartitions("timestamp")
                  .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = base.count()  # materialize
    logger('Base data materialized')

    # Calculate all features using the full batch (with past/future context)
    logger('Calculating basic features...')
    df = calculate_mid_prices(base).select("timestamp","trading_day","trading_hour","bids","asks","mid_price")
    df = calculate_log_returns(df).select("timestamp","trading_day","trading_hour","bids","asks","mid_price","log_return")
    
    # Forward returns (TARGETS - need future data)
    logger('Calculating forward returns (targets)...')
    df = derive_midprice_targets(df)
    
    # Past returns (FEATURES - need past data)
    logger('Calculating past returns (features)...')
    df = derive_midprice_past_returns(df)
    
    # Other microstructure features
    logger('Calculating variance and volatility...')
    df = estimate_variance(df, half_life=20)
    df = calculate_volatility(df)
    
    logger('Calculating spread and microprice...')
    df = calculate_spread(df)
    df = calculate_microprice(df)
    
    # Depth-based features
    logger('Calculating depth-based features...')
    BANDS_K = [5, 15, 50, 0]  # very-near, near, middle, whole
    
    for kk in BANDS_K:
        df = calculate_depth_imbalance(df, k=kk)
        df = calculate_market_depth(df, k=kk)
        df = calculate_liquidity_concentration(df, k=kk)
        df = calculate_price_impact_proxy(df, k=kk)
        df = calculate_liquidity_spread(df, k=kk)

    # Build column selection
    keep = ["timestamp","trading_day","trading_hour","bids","asks","mid_price","log_return","variance_proxy","volatility","spread","microprice"]
    keep += [c for c in df.columns if c.startswith("fwd_logret_")]    # Forward returns (targets)
    keep += [c for c in df.columns if c.startswith("past_logret_")]   # Past returns (features)
    keep += [c for c in df.columns if c.startswith("depth_imbalance_")]
    keep += [c for c in df.columns if c.startswith("depth_")]
    keep += [c for c in df.columns if c.startswith("liquidity_concentration_")]
    keep += [c for c in df.columns if c.startswith("price_impact_proxy_")]
    keep += [c for c in df.columns if c.startswith("liquidity_spread_")]

    df = df.select(*keep)
    
    # CRITICAL: Filter to ONLY the target hour after all calculations
    df_target_only = df.filter(col("trading_hour") == lit(target_hour))
    
    cnt = df_target_only.count()
    logger(f'Feature engineering complete - target hour {target_hour}, rows={cnt}')
    
    base.unpersist()
    return df_target_only

# ==============================
# Stage 2 Batch Processing
# ==============================

def write_micro_repartition(df: DataFrame, target_rows_per_part: int = 50) -> None:
    logger('Writing features to MongoDB...')
    total = df.count()
    logger(f'Total rows to write: {total}')

    # Coalesce to single partition for small batches
    df2 = df.coalesce(1)

    t0 = time.time()
    write_features_to_mongo(df2)
    logger(f'Write completed in {time.time() - t0:.2f}s')

def process_stage2_hourly_batches(raw_lob: DataFrame, limit_hours: int = None) -> None:
    """
    STAGE 2: Process raw LOB data into feature-engineered data with proper time windows.
    """
    logger('STAGE 2 FEATURE ENGINEERING: START')
    
    # Get all processable hours
    processable_hours = get_processable_hours(raw_lob, required_past_hours=3, required_future_hours=3)
    
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
        
        # Load batch with proper context window
        hour_batch = load_hour_batch(raw_lob, target_hour, required_past_hours=3, required_future_hours=3)
        
        batch_count = hour_batch.count()
        logger(f'Loaded batch for target hour {target_hour} - {batch_count} rows')
        
        if batch_count > 0:
            # Process features
            features_df = derive_hour_features(hour_batch, target_hour)
            
            # Verify correct hour
            processed_hours = (features_df.select("trading_hour").distinct().collect())
            actual_hours = [row.trading_hour for row in processed_hours]
            
            if len(actual_hours) == 1 and actual_hours[0] == target_hour:
                logger(f'✓ Correctly processed target hour: {target_hour}')
            else:
                logger(f'✗ ERROR - Expected {target_hour}, got {actual_hours}')
            
            # Write to features collection
            write_micro_repartition(features_df, target_rows_per_part=50)
            
            processed_count = features_df.count()
            total_processed += processed_count
            
            features_df.unpersist()
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
    
    logger(f'STAGE 2 COMPLETED - Total processed: {total_processed} rows in {total_time:.2f}s')

# ==============================
# Main Stage 2 Execution
# ==============================

if __name__ == "__main__":
    start_time = time.time()
    
    logger('STAGE 2: Feature Engineering Pipeline')
    logger(f'Input: {RAW_LOB_COLLECTION}')
    logger(f'Output: {FEATURES_LOB_COLLECTION}')
    
    # Load raw data from Stage 1
    logger('Loading raw LOB data from Stage 1...')
    raw_lob = load_raw_lob(limit=1000)  # Limit for testing, remove for full processing
    
    # Process features in hourly batches
    logger('Processing features in hourly batches...')
    process_stage2_hourly_batches(raw_lob, limit_hours=3)  # Process first 3 hours for testing
    
    total_time = time.time() - start_time
    logger(f'STAGE 2 completed in {total_time:.2f} seconds')
    logger('Next: Run Stage 3 (Standardization) to process features_lob -> standard_lob')
    
    spark.stop()