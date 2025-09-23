import math
import sys, time
from datetime import datetime
import numpy as np
from pyspark import StorageLevel
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import LongType
from pyspark.sql.functions import (
    col, expr, log, lag, when, to_date, pow, row_number, lit, sum as spark_sum, date_trunc, approx_count_distinct
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
LANDING_TRADE = "landing_trade"
LANDING_LOB = "landing_lob"
FORMATTED_TRADE = "formatting_trade"
FORMATTED_LOB = "formatting_lob"

spark = (
    SparkSession.builder
    .appName("FormattedZone")
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

    # Keep bulks tiny â€" mirrors the stable ~50-sample behavior
    .config("spark.mongodb.write.ordered", "false")
    .config("spark.mongodb.write.writeConcern.w", "1")

    # Timeouts / robustness
    .config("spark.mongodb.connection.timeout.ms", "30000")
    .config("spark.mongodb.socket.timeout.ms", "120000")
    .config("spark.mongodb.write.retryWrites", "true")

    # Memory / network tolerance
    .config("spark.driver.memory", "6g")
    .config("spark.network.timeout", "300s")
    .config("spark.executor.heartbeatInterval", "60s")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir("file:///C:/tmp/spark_checkpoints")

# ==============================
# IO
# ==============================

def logger(msg: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def load_lob() -> DataFrame:
    """
    Loads Landing LOB with projection pushdown (timestamp/bids/asks only).
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
    Returns list of (trading_hour) values.
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
    If include_overlap=True, includes some data from previous hour for continuity in calculations.
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
# Feature derivation (BATCH-LOCAL, SQL-only)
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
    v_t = α * β^{t} * sum_{i=0..t} (r_i^2 * β^{-i}), with β = 1-α, α = 1 - 2^(-1/half_life)
    """
    alpha = 1.0 - pow(lit(2.0), lit(-1.0) / lit(float(half_life)))  # α = 1 - 2^(-1/half_life)
    beta  = (lit(1.0) - alpha)                                      # β = 1 - α

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

def derive_hour_features_sql(hour_batch: DataFrame) -> DataFrame:
    """
    Process features for a single hour batch with time-ordered SQL pipeline.
    """
    logger('derive_hour_features_sql: START')
    
    base = (
        hour_batch.select("timestamp","bids","asks")
                  .withColumn("trading_day", to_date(col("timestamp")))
                  .withColumn("trading_hour", date_trunc("hour", col("timestamp")))
                  .repartition(4)  # smaller partitions for hour batches
                  .sortWithinPartitions("timestamp")
                  .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = base.count()  # materialize once
    logger('derive_hour_features_sql: base materialized')

    logger('derive_hour_features_sql: mid_price...')
    df = calculate_mid_prices(base).select("timestamp","trading_day","trading_hour","bids","asks","mid_price")
    logger('derive_hour_features_sql: log_return...')
    df = calculate_log_returns(df).select("timestamp","trading_day","trading_hour","bids","asks","mid_price","log_return")
    logger('derive_hour_features_sql: variance...')
    df = estimate_variance(df, half_life=20).select("timestamp","trading_day","trading_hour","bids","asks",
                                                    "mid_price","log_return","variance_proxy")
    logger('derive_hour_features_sql: volatility...')
    df = calculate_volatility(df).select("timestamp","trading_day","trading_hour","bids","asks",
                                         "mid_price","log_return","variance_proxy","volatility")
    logger('derive_hour_features_sql: spread...')
    df = calculate_spread(df).select("timestamp","trading_day","trading_hour","bids","asks",
                                     "mid_price","log_return","variance_proxy","volatility","spread")
    
    # ---- Banded features by top-k levels from mid (0 => whole book) ----
    BANDS_K = [5, 15, 50, 0]  # very-near, near, middle, whole
    
    logger('derive_hour_features_sql: banded features (depth_imbalance, depth, liq_conc, price_impact, liq_spread)...')
    for kk in BANDS_K:
        df = calculate_depth_imbalance(df, k=kk)
        df = calculate_market_depth(df, k=kk)
        df = calculate_liquidity_concentration(df, k=kk)
        df = calculate_price_impact_proxy(df, k=kk)
        df = calculate_liquidity_spread(df, k=kk)

    logger('derive_hour_features_sql: microprice...')
    df = calculate_microprice(df)

    # Build keep list
    keep = ["timestamp","trading_day","trading_hour","bids","asks","mid_price","log_return","variance_proxy","volatility","spread"]
    keep = list(set(keep) | {"microprice"})

    # Collect all generated band columns
    keep += [c for c in df.columns if c.startswith("depth_imbalance_")]
    keep += [c for c in df.columns if c.startswith("depth_")]
    keep += [c for c in df.columns if c.startswith("liquidity_concentration_")]
    keep += [c for c in df.columns if c.startswith("price_impact_proxy_")]
    keep += [c for c in df.columns if c.startswith("liquidity_spread_")]

    df = df.select(*keep)
    
    # Filter out the overlap data (keep only current hour data for writing)
    current_hour_only = df.filter(
        col("trading_hour") == df.select("trading_hour").distinct().orderBy(col("trading_hour").desc()).first()[0]
    )

    cnt = current_hour_only.count()
    logger(f'derive_hour_features_sql: DONE features for hour, rows={cnt}')
    base.unpersist()
    return current_hour_only

# ==============================
# Batch processing and writing
# ==============================

def write_micro_repartition(df: DataFrame, target_rows_per_part: int = 50) -> None:
    logger('write_micro_repartition: START')
    total = df.count()
    logger(f'write_micro_repartition: total={total}')

    # Avoid shuffle/repartition entirely; write from a single small task.
    df2 = df.coalesce(1)

    t0 = time.time()
    logger('write_micro_repartition: writing (single partition)...')
    write_append_mongo(df2)
    logger(f'write_micro_repartition: DONE in {time.time() - t0:.2f}s')

def process_hourly_batches(lob_raw: DataFrame) -> None:
    """
    Process LOB data in hourly batches for memory efficiency.
    """
    logger('process_hourly_batches: START')
    
    # Get all hourly ranges
    hourly_ranges = get_hourly_ranges(lob_raw)
    logger(f'process_hourly_batches: Found {len(hourly_ranges)} hourly batches')
    
    total_processed = 0
    
    for i, hour_start in enumerate(hourly_ranges):
        logger(f'process_hourly_batches: Processing batch {i+1}/{len(hourly_ranges)} - {hour_start}')
        
        # Load hour batch with overlap for continuity
        hour_batch = load_hour_batch(lob_raw, hour_start, include_overlap=True)
        
        batch_count = hour_batch.count()
        logger(f'process_hourly_batches: Batch {i+1} loaded, rows={batch_count}')
        
        if batch_count == 0:
            logger(f'process_hourly_batches: Skipping empty batch {i+1}')
            continue
        
        # Process features for this hour
        hour_features = derive_hour_features_sql(hour_batch)
        
        # Write features
        write_micro_repartition(hour_features, target_rows_per_part=50)
        
        processed_count = hour_features.count()
        total_processed += processed_count
        
        logger(f'process_hourly_batches: Batch {i+1} completed, processed={processed_count}, total_so_far={total_processed}')
        
        # Clean up to free memory
        hour_batch.unpersist()
        hour_features.unpersist()
    
    logger(f'process_hourly_batches: COMPLETED all batches, total_processed={total_processed}')

# ==============================
# Main
# ==============================

if __name__ == "__main__":
    logger('MAIN: START - Hourly Batch Processing')
    
    # 1) Load all LOB metadata (for getting hour ranges)
    logger('MAIN: load_lob')
    lob_raw = load_lob()

    # 2) Process in hourly batches
    logger('MAIN: process_hourly_batches')
    process_hourly_batches(lob_raw)
    
    logger('MAIN: COMPLETED')