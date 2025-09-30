# =================================================================================================
# Imports                                                                                        
# =================================================================================================

import math
import sys, time
import numpy as np
from datetime import datetime
from pyspark import StorageLevel
from pyspark.sql.window import Window
from pyspark.sql.types import LongType
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, expr, log, lag, lead, when, to_date, pow, row_number, lit, sum as spark_sum, 
    date_trunc, approx_count_distinct
)

# =================================================================================================
# Settings                                                                                         
# =================================================================================================

jar_files_path = "file:///C:/Users/llucp/spark_jars/"

jar_files = [
    "mongo-spark-connector_2.12-10.1.1.jar",
    "mongodb-driver-core-4.10.1.jar",
    "mongodb-driver-sync-4.10.1.jar",
    "bson-4.10.1.jar"
]

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "tfg"
RAW_LOB_COLLECTION = "raw_lob_train"
FEATURES_LOB_COLLECTION = "features_lob_train"

spark = (
    SparkSession.builder
    .appName("Stage2_FeatureEngineering")
    .config("spark.jars", ",".join([jar_files_path + jar for jar in jar_files]))
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "512")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.mongodb.read.connection.uri", MONGO_URI)
    .config("spark.mongodb.write.connection.uri", MONGO_URI)
    .config("spark.mongodb.read.database", DB_NAME)
    .config("spark.mongodb.write.database", DB_NAME)
    .config("spark.mongodb.write.ordered", "false")
    .config("spark.mongodb.write.writeConcern.w", "1")
    .config("spark.mongodb.connection.timeout.ms", "30000")
    .config("spark.mongodb.socket.timeout.ms", "120000")
    .config("spark.mongodb.write.retryWrites", "true")
    .config("spark.driver.memory", "8g")
    .config("spark.network.timeout", "300s")
    .config("spark.executor.heartbeatInterval", "60s")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# =================================================================================================
# Utilities                                                                                    
# =================================================================================================

def logger(msg: str) -> None:
    """
    Shows workflow messages while executing script through terminal.
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[LOB FEATURE DERIVATION] [{ts}] {msg}")
    sys.stdout.flush()

def load_raw_lob() -> DataFrame:
    """
    Loads raw LOB data from the provided database collection.
    """
    logger(f'Loading raw LOB data from {RAW_LOB_COLLECTION}')
    
    pipeline = [
        {"$sort": {"timestamp": 1}},
        {"$project": {"timestamp": 1, "bids": 1, "asks": 1}}
    ]
    
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

def write_to_database(df: DataFrame) -> None:
    """
    Writes preprocessed data to the provided database collection.
    """
    (df.write.format("mongodb")
       .option("database", DB_NAME)
       .option("collection", FEATURES_LOB_COLLECTION)
       .option("ordered", "false")
       .mode("append")
       .save())
    
# =================================================================================================
# Forward returns derivation                                                           
# =================================================================================================

def calculate_forward_log_returns(df: DataFrame, lag: int, N: int) -> DataFrame:
    """
    Calculates forward log-returns of mid-price, accounting with a decision lag.
    """
    w = Window.orderBy("timestamp")
    base = lead(col("mid_price"), lag).over(w)
    future = lead(col("mid_price"), lag + N).over(w)

    return df.withColumn(f"fwd_logret_{N}", log(future) - log(base))

def derive_forward_log_returns(df: DataFrame) -> DataFrame:
    H = [2, 3, 4, 5, 10, 20, 40, 60, 120, 240]
    lag = 1

    result = df
    for N in H:
        logger(f'Calculating fwd_logret_{N}...')
        result = calculate_forward_log_returns(result, lag=lag, N=N)

    return result

# =================================================================================================
# LOB features derivation                                                                   
# =================================================================================================

def calculate_historical_log_returns(df: DataFrame, N: int) -> DataFrame:
    """
    Calculates historical log returns.
    """
    w = Window.orderBy("timestamp")
    past = lag(col("mid_price"), N).over(w)

    return df.withColumn(f"past_logret_{N}", log(col("mid_price")) - log(past))

def derive_historical_returns(df: DataFrame) -> DataFrame:
    H = [1, 2, 3, 4, 5, 10, 20, 40, 60, 120, 240]

    result = df

    for N in H:
        logger(f'Calculating past_logret_{N}...')
        result = calculate_historical_log_returns(result, N=N)

    return result

def calculate_mid_prices(df: DataFrame) -> DataFrame:
    """
    Calculate mid-price from LOB data.
    Uses aggregate with greatest/least to find best bid/ask regardless of sorting.
    """
    best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
    best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")
    return df.withColumn("mid_price", (best_bid + best_ask) / 2.0)

def calculate_log_returns(df: DataFrame) -> DataFrame:
    w = Window.orderBy("timestamp")
    return df.withColumn("log_return", log(col("mid_price")) - log(lag(col("mid_price"), 1).over(w)))

def estimate_variance(df: DataFrame, half_life: int) -> DataFrame:
    """
    Calculates a variance estimator of log-returns using an EWMA approach.
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

def calculate_volatility(df: DataFrame) -> DataFrame:
    """
    Calculates an estimate of volatility using the variance estimator.
    """
    return df.withColumn("volatility", expr("sqrt(variance_proxy)"))

def calculate_spread(df: DataFrame) -> DataFrame:
    """
    Calculates bid-ask spread.
    """
    best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
    best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")
    return df.withColumn("spread", best_ask - best_bid)

def calculate_microprice(df: DataFrame) -> DataFrame:
    """
    Calculates volume-weighted microprice.
    """
    # Finds best prices
    best_bid = expr("aggregate(bids, CAST(-1.0E308 AS DOUBLE), (acc,x) -> greatest(acc, x[0]))")
    best_ask = expr("aggregate(asks, CAST( 1.0E308 AS DOUBLE), (acc,x) -> least(acc, x[0]))")
    df = df.withColumn("best_bid", best_bid).withColumn("best_ask", best_ask)

    # Sums volumes at best prices
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
    """
    Calculates order book depth imbalance for top-k levels.
    """
    K = int(k)
    
    # Converts to struct and sorts to get top of book first
    b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
    a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
    b_sorted = f"reverse(array_sort({b_struct}))"
    a_sorted = f"array_sort({a_struct})"
    
    # Selects top k levels
    b_topk = f"slice({b_sorted}, 1, {K})" if K > 0 else b_sorted
    a_topk = f"slice({a_sorted}, 1, {K})" if K > 0 else a_sorted
    
    # Adds up volumes
    vb = f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y) -> acc + y)"
    va = f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y) -> acc + y)"
    
    # Calculates imbalance
    denom = f"(({vb}) + ({va}))"
    imb = f"CASE WHEN ({denom})=0D THEN 0D ELSE (({vb}) - ({va})) / ({denom}) END"

    suffix = f"top_{K}" if K > 0 else "all"

    return df.withColumn(f"depth_imbalance_{suffix}", expr(imb))

def calculate_market_depth(df: DataFrame, k: int) -> DataFrame:
    """
    Calculate total market depth for top-k levels.
    """
    K = int(k)
    
    # Converts to struct and sorts to get top of book first
    b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
    a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
    b_sorted = f"reverse(array_sort({b_struct}))"
    a_sorted = f"array_sort({a_struct})"
    
    # Selects top k levels
    b_topk = f"slice({b_sorted}, 1, {K})" if K > 0 else b_sorted
    a_topk = f"slice({a_sorted}, 1, {K})" if K > 0 else a_sorted
    
    # Adds up all volumes
    depth_sum = (
        f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y)->acc+y) + "
        f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y)->acc+y)"
    )

    suffix = f"top_{K}" if K > 0 else "all"

    return df.withColumn(f"depth_{suffix}", expr(depth_sum))

def calculate_liquidity_concentration(df: DataFrame, k: int) -> DataFrame:
    """
    Calculates liquidity concentration for top-k levels
    """
    K = int(k)
    
    # Converts to struct and sorts to get top of book first
    b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
    a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
    b_sorted = f"reverse(array_sort({b_struct}))"
    a_sorted = f"array_sort({a_struct})"
    
    # Selects top-k levels
    b_topk = f"slice({b_sorted}, 1, {K})" if K > 0 else b_sorted
    a_topk = f"slice({a_sorted}, 1, {K})" if K > 0 else a_sorted
    
    # Adds up all volumes
    denom = (
        f"aggregate(transform({b_sorted}, x -> x.v), 0D, (acc,y)->acc+y) + "
        f"aggregate(transform({a_sorted}, x -> x.v), 0D, (acc,y)->acc+y)"
    )
    
    # Adds up just top-k volumes
    num = (
        f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y)->acc+y) + "
        f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y)->acc+y)"
    )
    
    # Calculates liquidity concentration
    lc = f"CASE WHEN ({denom})=0D THEN 0D ELSE ({num})/({denom}) END"
    
    suffix = f"top_{K}" if K > 0 else "all"

    return df.withColumn(f"liquidity_concentration_{suffix}", expr(lc))

def calculate_price_impact_proxy(df: DataFrame, k: int, eps: float = 1e-12) -> DataFrame:
    """
    Calculates price impact for top-k levels.
    """
    K = int(k)
    
    # Converts to struct and sorts to get top of book first
    b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
    a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
    b_sorted = f"reverse(array_sort({b_struct}))"
    a_sorted = f"array_sort({a_struct})"
    
    # Selects price at k level 
    b_eff = f"CASE WHEN {K} > 0 THEN least({K}, size({b_sorted})) ELSE size({b_sorted}) END"
    a_eff = f"CASE WHEN {K} > 0 THEN least({K}, size({a_sorted})) ELSE size({a_sorted}) END"   
    PbK = f"element_at({b_sorted}, {b_eff}).p"
    PaK = f"element_at({a_sorted}, {a_eff}).p"
    
    # Selects top-k price levels
    b_topk = f"slice({b_sorted}, 1, {b_eff})"
    a_topk = f"slice({a_sorted}, 1, {a_eff})"
    
    # Adds up just top-k volumes
    depth_sum = (
        f"aggregate(transform({b_topk}, x -> x.v), 0D, (acc,y)->acc+y) + "
        f"aggregate(transform({a_topk}, x -> x.v), 0D, (acc,y)->acc+y)"
    )
    
    # Calculates price impact proxy
    lam = f"CASE WHEN ({depth_sum})<=0D THEN 0D ELSE (({PaK}) - ({PbK})) / ({depth_sum} + {eps}) END"
    
    suffix = f"top_{K}" if K > 0 else "all"

    return df.withColumn(f"price_impact_proxy_{suffix}", expr(lam))

def calculate_liquidity_spread(df: DataFrame, k: int, eps: float = 1e-12) -> DataFrame:
    """
    Calculates liquidity spread for top k-levels.
    """
    K = int(k)
    
    # Converts to struct and sorts to get top of book first
    b_struct = "transform(bids, x -> named_struct('p', x[0], 'v', x[1]))"
    a_struct = "transform(asks, x -> named_struct('p', x[0], 'v', x[1]))"
    b_sorted = f"reverse(array_sort({b_struct}))"
    a_sorted = f"array_sort({a_struct})"
    
    # Selects prices at top-k level 
    kmin_all = f"least(size({b_sorted}), size({a_sorted}))"
    k_eff = f"CASE WHEN {K} > 0 THEN least({K}, {kmin_all}) ELSE {kmin_all} END"
    b_top = f"slice({b_sorted}, 1, {k_eff})"
    a_top = f"slice({a_sorted}, 1, {k_eff})"
    
    # Extracts prices and volumes for top-k levels
    Pa = f"transform({a_top}, x -> x.p)"
    Va = f"transform({a_top}, x -> x.v)"
    Pb = f"transform({b_top}, x -> x.p)"
    Vb = f"transform({b_top}, x -> x.v)"
    
    # Calculates weights and differences
    w = f"zip_with({Va}, {Vb}, (va, vb) -> va + vb)"
    diff = f"zip_with({Pa}, {Pb}, (pa, pb) -> pa - pb)"
    
    # Calculates liquidity spread
    num_arr = f"zip_with({diff}, {w}, (d, ww) -> d * ww)"
    denom = f"aggregate({w}, 0D, (acc, y) -> acc + y)"
    num   = f"aggregate({num_arr}, 0D, (acc, y) -> acc + y)"
    lspread = f"CASE WHEN ({denom})<=0D THEN 0D ELSE ({num}) / ({denom} + {eps}) END"

    suffix = f"top_{K}" if K > 0 else "all"

    return df.withColumn(f"liquidity_spread_{suffix}", expr(lspread))

# =================================================================================================
# Pipeline                                                                                        
# =================================================================================================

def get_preprocessable_hours(df: DataFrame, required_past_hours: int = 3, required_future_hours: int = 3) -> list:
    """
    Given all raw LOB data and considering the preprocessing pipeline needs, returns the hours that can be
    preprocessed fully by the pipeline. This excludes samples used in calculating first and last feature
    of the sample period.
    """
    all_hours = (df.withColumn("trading_hour", date_trunc("hour", col("timestamp")))
                   .select("trading_hour")
                   .distinct()
                   .orderBy("trading_hour")
                   .collect())
    
    available_hours = [row.trading_hour for row in all_hours]

    logger(f'Available hours in database: {len(available_hours)} hours from {available_hours[0]} to {available_hours[-1]}')
    
    processable_hours = []

    for i, hour in enumerate(available_hours):
        past_hours_available = i
        future_hours_available = len(available_hours) - i - 1
        
        if past_hours_available >= required_past_hours and future_hours_available >= required_future_hours:
            processable_hours.append(hour)
    
    logger(f'Preprocessable hours: {len(processable_hours)} hours')

    if processable_hours:
        logger(f'Range: {processable_hours[0]} to {processable_hours[-1]}')
    
    return processable_hours

def load_hour_batch(df: DataFrame, target_hour, required_past_hours: int = 3, required_future_hours: int = 3):
    """
    Loads batches of data considering preprocessing pipeline needs.
    """
    past_start = expr(f"'{target_hour}' - INTERVAL {required_past_hours} HOURS")
    future_end = expr(f"'{target_hour}' + INTERVAL {required_future_hours + 1} HOURS")
    
    filter_condition = (
        (col("timestamp") >= past_start) &
        (col("timestamp") < future_end)
    )
    
    logger(f'Loading batch for target hour {target_hour}')
    logger(f'Window: {target_hour} - {required_past_hours}h to {target_hour} + {required_future_hours + 1}h')
    
    return df.filter(filter_condition)

def derive_hour_features(hour_batch: DataFrame, target_hour, is_first_batch: bool = False) -> DataFrame:
    """
    Given an hour batch of data preprocesses it by deriving LOB features and target.
    """
    logger(f'Deriving LOB features for target hour: {target_hour} (is_first_batch={is_first_batch})')
    
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
    
    base = (
        hour_batch.select("timestamp", "bids", "asks")
                  .withColumn("trading_hour", date_trunc("hour", col("timestamp")))
                  .repartition(4)
                  .sortWithinPartitions("timestamp")
                  .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = base.count()

    logger('Calculating price fundamental features...')
    df = calculate_mid_prices(base).select("timestamp", "trading_hour", "bids", "asks", "mid_price")
    df = calculate_log_returns(df).select("timestamp", "trading_hour", "bids", "asks", "mid_price", "log_return")
    
    logger('Calculating forward returns (targets)...')
    df = derive_forward_log_returns(df)
    
    logger('Calculating historical returns (features)...')
    df = derive_historical_returns(df)
    
    logger('Calculating variance and volatility...')
    df = estimate_variance(df, half_life=20)
    df = calculate_volatility(df)
    
    logger('Calculating spread and microprice...')
    df = calculate_spread(df)
    df = calculate_microprice(df)
    
    logger('Calculating depth-based features...')
    BANDS_K = [5, 15, 50, 0]  # very-near, near, middle, whole
    
    for kk in BANDS_K:
        df = calculate_depth_imbalance(df, k=kk)
        df = calculate_market_depth(df, k=kk)
        df = calculate_liquidity_concentration(df, k=kk)
        df = calculate_price_impact_proxy(df, k=kk)
        df = calculate_liquidity_spread(df, k=kk)

    keep = ["timestamp", "trading_hour", "bids", "asks", "mid_price", "log_return", "variance_proxy", "volatility", "spread", "microprice"]
    keep += [c for c in df.columns if c.startswith("fwd_logret_")]
    keep += [c for c in df.columns if c.startswith("past_logret_")]
    keep += [c for c in df.columns if c.startswith("depth_imbalance_")]
    keep += [c for c in df.columns if c.startswith("depth_")]
    keep += [c for c in df.columns if c.startswith("liquidity_concentration_")]
    keep += [c for c in df.columns if c.startswith("price_impact_proxy_")]
    keep += [c for c in df.columns if c.startswith("liquidity_spread_")]

    df = df.select(*keep)
    
    # Adds is_context flag: mark rows as context (for estimator or tagret warm-up)
    df = df.withColumn("is_context", when(col("trading_hour") < lit(target_hour), lit(True)).otherwise(lit(False)))
    
    if is_first_batch:
        # For the first batch, keeps context
        context_end = expr(f"'{target_hour}' + INTERVAL 1 HOURS")
        df_output = df.filter(col("timestamp") < context_end)
        context_cnt = df_output.filter(col("is_context") == True).count()
        target_cnt = df_output.filter(col("is_context") == False).count()
        logger(f'First batch - keeping context hours: {context_cnt} context rows, {target_cnt} target rows')
        
    else:
        # For subsequent batches, only keep target hour (context was already saved)
        df_output = df.filter(col("trading_hour") == lit(target_hour))
        target_cnt = df_output.count()
        logger(f'Regular batch - keeping only target hour: {target_cnt} target rows')
    
    cnt = df_output.count()
    logger(f'LOB feature derivation complete - target hour {target_hour}, rows={cnt}')
    base.unpersist()

    return df_output

def preprocess_stage4_hour_batches(raw_lob: DataFrame) -> None:
    """
    Preprocessing pipeline orchestrator.
    """
    logger('STAGE 4 LOB FEATURE DERIVATION: START')
    processable_hours = get_preprocessable_hours(raw_lob, required_past_hours=3, required_future_hours=3)
    
    if not processable_hours:
        logger('ERROR - No processable hours found!')
        return
    
    logger(f'Processing {len(processable_hours)} hours')
    logger(f'First processable hour: {processable_hours[0]}')
    logger(f'Last processable hour: {processable_hours[-1]}')
    total_processed = 0
    total_time = 0
    
    for i, target_hour in enumerate(processable_hours):
        batch_start_time = time.time()
        is_first = (i == 0)  # First processable hour needs context preservation
        logger(f'Processing {i+1}/{len(processable_hours)} - TARGET HOUR: {target_hour} (first_batch={is_first})')
        hour_batch = load_hour_batch(raw_lob, target_hour, required_past_hours=3, required_future_hours=3)
        batch_count = hour_batch.count()
        logger(f'Loaded batch for target hour {target_hour} - {batch_count} rows')
        
        if batch_count > 0:
            features_df = derive_hour_features(hour_batch, target_hour, is_first_batch=is_first)
            processed_hours = (features_df.select("trading_hour").distinct().collect())
            actual_hours = [row.trading_hour for row in processed_hours]
            actual_hours.sort()
            
            if is_first:
                logger(f'First batch processed - hours: {actual_hours}')

            else:
                if len(actual_hours) == 1 and actual_hours[0] == target_hour:
                    logger(f'Correctly processed target hour: {target_hour}')

                else:
                    logger(f'ERROR - Expected {target_hour}, got {actual_hours}')
            
            # Writting preprocessed batch to database
            logger('Writing LOB derived features to database...')
            total = features_df.count()
            logger(f'Total rows to write: {total}')
            t0 = time.time()
            write_to_database(features_df)
            logger(f'Write completed in {time.time() - t0:.2f}s')

            processed_count = features_df.count()
            total_processed += processed_count
            features_df.unpersist()

        else:
            logger(f'Skipping empty batch for {target_hour}')
        
        # Runtime statistics management
        batch_duration = time.time() - batch_start_time
        total_time += batch_duration
        avg_time_per_batch = total_time / (i + 1)
        estimated_remaining = avg_time_per_batch * (len(processable_hours) - i - 1)
        logger(f'Batch completed in {batch_duration:.2f}s')
        logger(f'Progress: {i+1}/{len(processable_hours)}, ETA: {estimated_remaining:.2f}s')

        hour_batch.unpersist()
    
    logger(f'STAGE 4 COMPLETED - Total processed: {total_processed} rows in {total_time:.2f}s')

# =================================================================================================
# Main                                                                                       
# =================================================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    logger('STAGE 4: LOB FEATURE DERIVATION PIPELINE')
    logger(f'Source database collection: {RAW_LOB_COLLECTION}')
    logger(f'Target database collection: {FEATURES_LOB_COLLECTION}')

    logger('Loading raw LOB data from STAGE 3...')
    raw_lob = load_raw_lob()
    
    logger('Deriving LOB features in hourly batches...')
    preprocess_stage4_hour_batches(raw_lob)
    
    total_time = time.time() - start_time
    logger(f'STAGE 4 completed in {total_time:.2f} seconds')