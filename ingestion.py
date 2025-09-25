import os
import time
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType, TimestampType, StringType, IntegerType

# Initializes Spark session with MongoDB connector
jar_files_path = "file:///C:/Users/llucp/spark_jars/"

jar_files = [
    "mongo-spark-connector_2.12-10.1.1.jar",
    "mongodb-driver-core-4.10.1.jar",
    "mongodb-driver-sync-4.10.1.jar",
    "bson-4.10.1.jar"
]

# Data paths and MongoDB configuration - UPDATED FOR NEW ARCHITECTURE
LOB_DATA = "./lob_data"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tfg"

# STAGE 1: RAW INGESTION COLLECTIONS  
RAW_LOB_COLL = "raw_lob"         # Changed from "landing"
INGESTION_LOG_COLL = "ingestion_log"  # Changed from "log"

spark = (
    SparkSession.builder
    .appName("Stage1_RawIngestion")    # Updated app name
    .config("spark.jars", ",".join([jar_files_path + jar for jar in jar_files]))
    .config("spark.mongodb.connection.uri", MONGO_URI)
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

def already_uploaded_files(label: str) -> set:
    """
    Loads the prior uploaded files log from MongoDB and returns a set of filenames.
    """
    try:
        log_df = (
            spark.read.format("mongodb")
            .option("database", DB_NAME)
            .option("collection", INGESTION_LOG_COLL)
            .load()
            .select("filename", "type")
        )

        uploaded = (
            log_df.filter(log_df["type"] == label)
            .select("filename")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        return set(uploaded)
    
    except Exception:
        # If the collection doesn't exist yet, treats it as an empty set
        print(f"[WARN] No prior log found for {label}. Starting fresh.")
        return set()

def parse_lob(df: DataFrame) -> DataFrame:
    """
    Parses LOB data: converts timestamp to TimestampType and JSON strings to arrays.
    STAGE 1: Basic parsing and type conversion only.
    """    
    # Define array structure for bids/asks: array of [price, volume] pairs
    price_volume_pair = ArrayType(ArrayType(DoubleType()))

    return (
        df
        .withColumn("timestamp", col("timestamp").cast(TimestampType()))
        .withColumn("bids", from_json(col("bids").cast("string"), price_volume_pair))
        .withColumn("asks", from_json(col("asks").cast("string"), price_volume_pair))
    )

def update_ingestion_log(filename: str, label: str, count: int) -> None:
    """
    Updates ingestion log to keep track of raw data uploads.
    Uses the same structure as your working landing script.
    """
    schema = StructType([
        StructField("filename", StringType(), False),
        StructField("type", StringType(), False),
        StructField("record_count", IntegerType(), False),
    ])

    log_row_df = spark.createDataFrame([(filename, label, int(count))], schema=schema)

    (
        log_row_df.write.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", INGESTION_LOG_COLL)
        .mode("append")
        .save()
    )

def ingest_raw_lob_data() -> None:
    """
    STAGE 1: Ingests raw LOB Parquet files into MongoDB raw collection.
    Based on your working landing script structure.
    """
    if not os.path.isdir(LOB_DATA):
        print(f"[WARN] LOB data folder not found: {LOB_DATA}. No data to ingest.")
        return
    
    # Get list of already processed files
    already_processed = already_uploaded_files("Raw LOB")
    files = sorted([f for f in os.listdir(LOB_DATA) if f.endswith(".parquet")])
    
    if not files:
        print(f"[WARN] No Parquet files found in {LOB_DATA}")
        return
    
    print(f"[INFO] STAGE 1 INGESTION: Found {len(files)} Parquet files, {len(already_processed)} already processed")
    
    total_records = 0
    processed_files = 0

    for file in files:
        if file in already_processed:
            print(f"[INFO] Skipping already processed file: {file}")
            continue

        full_path = os.path.join(LOB_DATA, file)
        print(f"[INFO] Processing: {file}")

        try:
            # Read and parse LOB data (same as your working version)
            df = spark.read.parquet(full_path).dropna()
            df = parse_lob(df)
            
            # Check if file has data
            record_count = df.count()
            if record_count == 0:
                print(f"[WARN] Empty file: {file}")
                update_ingestion_log(file, "Raw LOB", 0)
                continue

            # Upload to MongoDB RAW collection
            (
                df.write.format("mongodb")
                .option("database", DB_NAME)
                .option("collection", RAW_LOB_COLL)
                .option("replaceDocument", "false")
                .mode("append")
                .save()
            )

            # Update log and counters
            update_ingestion_log(file, "Raw LOB", record_count)
            total_records += record_count
            processed_files += 1
            
            print(f"[INFO] ✓ {file} -> {record_count} records uploaded to {RAW_LOB_COLL}")
        
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")
            # Continue with next file instead of stopping entire process

    print(f"[INFO] STAGE 1 INGESTION SUMMARY: {processed_files} files processed, {total_records} total records in {RAW_LOB_COLL}")

if __name__ == "__main__":
    start_time = time.time()
    
    print(f"[INFO] Starting STAGE 1: Raw LOB data ingestion to MongoDB")
    print(f"[INFO] Source directory: {os.path.abspath(LOB_DATA)}")
    print(f"[INFO] Target collection: {DB_NAME}.{RAW_LOB_COLL}")
    
    # Stage 1: Ingest raw LOB data
    ingest_raw_lob_data()
    
    end_time = time.time()
    print(f"[INFO] STAGE 1 ingestion completed in {end_time - start_time:.2f} seconds.")
    print(f"[INFO] Next: Run Stage 2 (Feature Engineering) to process {RAW_LOB_COLL} -> features_lob")
    
    spark.stop()