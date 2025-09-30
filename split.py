import sys
import time
from datetime import datetime
from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, date_trunc

# Initializes Spark session with MongoDB connector
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

spark = (
    SparkSession.builder
    .appName("Stage1_RawIngestion")    # Updated app name
    .config("spark.jars", ",".join([jar_files_path + jar for jar in jar_files]))
    .config("spark.mongodb.connection.uri", MONGO_URI)
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

def logger(msg: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[STAGE2] [{ts}] {msg}")
    sys.stdout.flush()

def load_raw_lob() -> DataFrame:
    """
    STAGE 2: Loads raw LOB data from Stage 1 output.
    """
    logger(f'Loading raw LOB data from {RAW_LOB_COLLECTION}')
    
    pipeline = [
        {"$sort": {"timestamp": 1}},  # Ensure chronological order
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

# ==============================
# SIMPLE SPLIT AND LOAD SCRIPT
# ==============================

def get_processable_hours(raw_lob: DataFrame, required_past_hours: int = 3, required_future_hours: int = 3) -> List[datetime]:
    """
    Get all hours that can have complete features + targets (accounting for feature derivation needs).
    """
    logger('Analyzing time range for splits...')
    
    # Get all available hours
    all_hours = (raw_lob.withColumn("trading_hour", date_trunc("hour", col("timestamp")))
                       .select("trading_hour")
                       .distinct()
                       .orderBy("trading_hour")
                       .collect())
    
    available_hours = [row.trading_hour for row in all_hours]
    logger(f'Available hours: {len(available_hours)} hours from {available_hours[0]} to {available_hours[-1]}')
    
    # Filter hours that have sufficient context for feature derivation
    processable_hours = []
    for i, hour in enumerate(available_hours):
        past_hours_available = i  # Hours before current
        future_hours_available = len(available_hours) - i - 1  # Hours after current
        
        if past_hours_available >= required_past_hours and future_hours_available >= required_future_hours:
            processable_hours.append(hour)
    
    logger(f'Processable hours (after accounting for feature derivation): {len(processable_hours)} hours')
    logger(f'Excluded: {required_past_hours} hours at start + {required_future_hours} hours at end')
    if processable_hours:
        logger(f'Processable range: {processable_hours[0]} to {processable_hours[-1]}')
    
    return processable_hours

def create_temporal_splits(processable_hours: List[datetime], train_pct: float = 0.8, test_pct: float = 0.2) -> dict:
    """
    Split processable hours chronologically into train and test sets.
    """
    n_hours = len(processable_hours)
    
    # Calculate split index
    train_end_idx = int(n_hours * train_pct)
    
    splits = {
        'train': processable_hours[:train_end_idx],
        'test': processable_hours[train_end_idx:]
    }
    
    # Log split information
    logger('TEMPORAL SPLITS:')
    logger('-' * 60)
    for split_name, hours in splits.items():
        n_split_hours = len(hours)
        pct = (n_split_hours / n_hours) * 100 if n_hours > 0 else 0
        start_time = hours[0] if hours else None
        end_time = hours[-1] if hours else None
        logger(f'{split_name.upper()}: {n_split_hours:3d} hours ({pct:5.1f}%) | {start_time} to {end_time}')
    
    return splits

def load_data_for_split_hours(raw_lob: DataFrame, split_hours: List[datetime], required_past_hours: int = 3, required_future_hours: int = 3) -> DataFrame:
    """
    Load raw data for specific hours, including extended context needed for feature derivation.
    """
    if not split_hours:
        return raw_lob.limit(0)  # Return empty DataFrame
    
    # Calculate extended time range needed for feature derivation
    from datetime import timedelta
    
    split_start = split_hours[0]
    split_end = split_hours[-1] + timedelta(hours=1)  # Include full last hour
    
    # Extend range for feature derivation context
    extended_start = split_start - timedelta(hours=required_past_hours)
    extended_end = split_end + timedelta(hours=required_future_hours)
    
    logger(f'Split target range: {split_start} to {split_end}')
    logger(f'Extended range (for features): {extended_start} to {extended_end}')
    
    # Filter raw data to extended range
    filtered_data = raw_lob.filter(
        (col("timestamp") >= extended_start) & 
        (col("timestamp") < extended_end)
    )
    
    count = filtered_data.count()
    logger(f'Loaded {count} records for split (including context)')
    
    return filtered_data

def save_split_to_collection(data: DataFrame, split_name: str):
    """
    Save split data to corresponding MongoDB collection.
    """
    collection_name = f"raw_lob_{split_name}"
    logger(f'Saving to collection: {collection_name}')
    
    (data.write.format("mongodb")
     .option("database", DB_NAME)
     .option("collection", collection_name)
     .option("ordered", "false")
     .mode("overwrite")  # Replace existing data
     .save())
    
    count = data.count()
    logger(f'Saved {count} records to {collection_name}')

def run_simple_split_and_load(train_pct: float = 0.8, test_pct: float = 0.2):
    """
    Simple script that:
    1. Loads raw data
    2. Gets time range and accounts for feature derivation needs
    3. Splits chronologically into train/test sets
    4. Loads each split (with context) to corresponding collection
    """
    logger('=' * 60)
    logger('SIMPLE SPLIT AND LOAD SCRIPT')
    logger('=' * 60)
    
    # 1. Load raw data
    logger('Step 1: Loading raw data...')
    raw_lob = load_raw_lob()  # Your existing function
    total_count = raw_lob.count()
    logger(f'Loaded {total_count} total records')
    
    # 2. Get processable time range (accounting for feature derivation)
    logger('Step 2: Analyzing time range...')
    processable_hours = get_processable_hours(raw_lob, required_past_hours=3, required_future_hours=3)
    
    if not processable_hours:
        logger('ERROR: No processable hours found!')
        return
    
    # 3. Create temporal splits
    logger('Step 3: Creating temporal splits...')
    splits = create_temporal_splits(processable_hours, train_pct, test_pct)
    
    # 4. Load and save each split
    logger('Step 4: Loading and saving splits...')
    for split_name, split_hours in splits.items():
        if split_hours:
            logger(f'Processing {split_name} split...')
            
            # Load data for this split (including context for feature derivation)
            split_data = load_data_for_split_hours(raw_lob, split_hours, 
                                                 required_past_hours=3, 
                                                 required_future_hours=3)
            
            # Save to collection
            save_split_to_collection(split_data, split_name)
            
            split_data.unpersist()
        else:
            logger(f'Skipping {split_name} split - no hours assigned')
    
    logger('=' * 60)
    logger('SPLIT AND LOAD COMPLETED')
    logger('Created collections:')
    logger('  - raw_lob_train   (training data with context)')
    logger('  - raw_lob_test    (test data with context)')
    logger('')
    logger('Next step: Run feature derivation on each collection separately')
    logger('=' * 60)

# ==============================
# USAGE
# ==============================

if __name__ == "__main__":
    start_time = time.time()
    
    # Run the simple split and load
    run_simple_split_and_load(
        train_pct=0.8, 
        test_pct=0.2     
    )
    
    total_time = time.time() - start_time
    logger(f'Completed in {total_time:.2f} seconds')
    
    spark.stop()