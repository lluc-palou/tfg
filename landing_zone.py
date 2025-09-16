import os
import time
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json, transform, when
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType, TimestampType, StringType, IntegerType

# Initializes Spark session with MongoDB connector
jar_files_path = "file:///C:/Users/llucp/spark_jars/"

jar_files = [
    "mongo-spark-connector_2.12-10.1.1.jar",
    "mongodb-driver-core-4.10.1.jar",
    "mongodb-driver-sync-4.10.1.jar",
    "bson-4.10.1.jar"
]

LOB_DATA = "./lob_data"
TRADE_DATA = "./trade_data"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tfg"
LOB_COLL = "landing_lob"
TRADE_COLL = "landing_trade"
LOG_COLL = "log"

spark = (
    SparkSession.builder
    .appName("LandingZone")
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
            .option("collection", LOG_COLL)
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
    Parses and casts LOB data types.
    """    
    pair_arr = ArrayType(ArrayType(DoubleType()))

    return (
        df
        .withColumn("timestamp", col("timestamp").cast(TimestampType()))
        .withColumn("bids", from_json(col("bids").cast("string"), pair_arr))
        .withColumn("asks", from_json(col("asks").cast("string"), pair_arr))
    )
        
def parse_trade(df: DataFrame) -> DataFrame:
    """
    Parses data, casts trade data types and changes column names.
    """
    return (
        df
        .withColumn("timestamp", col("timestamp").cast(TimestampType()))
        .withColumnRenamed("volume", "traded_volume")
        .withColumnRenamed("last_price", "last_traded_price")
    )

def update_log(filename: str, label: str, count: int) -> None:
    """
    Updates files log to keep track of the MongoDB content.
    """
    schema = StructType(
        [
            StructField("filename", StringType(), False),
            StructField("type", StringType(), False),
            StructField("record_count",  IntegerType(), False),
        ]
    )

    log_row_df = spark.createDataFrame([(filename, label, int(count))], schema=schema)

    (
        log_row_df.write.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", LOG_COLL)
        .mode("append")
        .save()
    )

def ingest(path: str, target_collection: str, label:str) -> None:
    """
    Ingests all Parquet files from the given folder into MongoDB, skipping those already logged.
    """
    if not os.path.isdir(path):
        print(f"[WARN] Folder not found: {path}. Skipping {label}.")
        return
    
    already = already_uploaded_files(label)
    files = sorted([f for f in os.listdir(path) if f.endswith(".parquet")])

    for file in files:
        if file in already:
            # Skips previously uploaded files
            continue

        full_path = os.path.join(path, file)

        try:
            df = spark.read.parquet(full_path).dropna()

            # Parses data according to data file nature
            if target_collection == LOB_COLL:
                df = parse_lob(df)

            elif target_collection == TRADE_COLL:
                df = parse_trade(df)
            
            else:
                # No parser defined for this collection, uses raw data
                print(f"[WARN] No parser defined for {target_collection}, using raw data.")

            if df.limit(1).count() == 0:
                update_log(file, label, 0)
                continue 

            # Uploads current file to database
            (
                df.write.format("mongodb")
                .option("database", DB_NAME)
                .option("collection", target_collection)
                .option("replaceDocument", "false")
                .mode("append")
                .save()
            )

            # Logs current file
            rec_count = df.count()
            update_log(file, label, rec_count)
            print(f"[INFO] {label}: {file} -> {rec_count} rows")
        
        except Exception as e:
            # Logs error while ingesting data and continues with next file
            print(f"[ERROR] Could not ingest {file} for {label}: {e}")

if __name__ == "__main__":
    # Ingests parsed LOB and Trade files into the Landing Zone
    start_time = time.time()
    ingest(LOB_DATA, LOB_COLL, "Landing LOB")
    ingest(TRADE_DATA, TRADE_COLL, "Landing Trade")
    end_time = time.time()
    print(f"[INFO] Ingestion completed in {end_time - start_time:.2f} seconds.")
    spark.stop()