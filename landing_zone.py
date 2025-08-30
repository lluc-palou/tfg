"""
Landing Zone
-------------
Reads local Parquet files (LOB and Trade Summary data), cleans them, and writes 
them directly to MongoDB using the MongoDB Spark Connector. It also maintains an
upload log in MongoDB to keep track of files (skip files that were already ingested).
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Initializes Spark session with MongoDB connector
jar_path = "file:///C:/Users/llucp/spark_jars/"

all_jars = [
    "mongo-spark-connector_2.12-10.1.1.jar",
    "mongodb-driver-core-4.10.1.jar",
    "mongodb-driver-sync-4.10.1.jar",
    "bson-4.10.1.jar"
]

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tfg"
LOB_COLL = "landing_lob"
TRADE_COLL = "landing_trade_summary"
LOG_COLL = "log"

spark = (
    SparkSession.builder
    .appName("LandingZone")
    .config("spark.jars", ",".join([jar_path + jar for jar in all_jars]))
    .config("spark.mongodb.write.connection.uri", MONGO_URI)
    .config("spark.mongodb.write.database", DB_NAME)
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Defines paths to local folders where colected data is stored
lob_folder = "./lob_data"
trade_folder = "./trade_data"

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
        return set()

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

def ingest_folder(folder_path: str, target_collection: str, label:str) -> None:
    """
    Ingests all Parquet files from the given folder into MongoDB, skipping those already logged.
    """
    if not os.path.isdir(folder_path):
        print(f"[WARN] Folder not found: {folder_path}. Skipping {label}.")
        return
    
    already = already_uploaded_files(label)
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".parquet")])

    for file in files:
        if file in already:
            # Skips previously uploaded files
            continue

        full_path = os.path.join(folder_path, file)

        try:
            df = spark.read.parquet(full_path)
            df_clean = df.dropna().withColumn("source_file", lit(file))

            # Computes number of non-NaN value records
            rec_count = df_clean.count()

            if rec_count == 0:
                update_log(file, label, 0)
                continue 

            # Uploads files to MongoDB
            (
                df_clean.write.format("mongodb")
                .option("database", DB_NAME)
                .option("collection", target_collection)
                .mode("append")
                .save()
            )

            # Logs file
            update_log(file, label, rec_count)
            print(f"[OK] {label}: {file} -> {rec_count} rows")
        
        except Exception as e:
            print(f"[ERROR] Could not ingest {file} for {label}: {e}")

if __name__ == "__main__":
    # Uploads LOB and Trade summary files into MongoDB
    ingest_folder(lob_folder, LOB_COLL, "Landing LOB")
    ingest_folder(trade_folder, TRADE_COLL, "Landing Trade Summary")
    print("All data is up to date")
    spark.stop()