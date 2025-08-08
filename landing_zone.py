import os
from tqdm import tqdm
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["tfg"]
lob_collection = db["landing_lob"]
trade_collection = db["landing_trade"]
upload_log = db["upload_log"]

# Initializes Spark session
spark = (
    SparkSession.builder
    .appName("LandingZone")
    .config("spark.driver.memory", "4g") # Take a look at memory consumption when using spark vectorized processing
    .getOrCreate()
)

# Defines paths
lob_folder = "./lob_data"
trade_folder = "./trade_data"

def already_uploaded(filename, dataset_type):
    """
    Checks if file was previously uploaded.
    """
    return upload_log.find_one({"filename": filename, "type": dataset_type}) is not None

def log_upload(filename, dataset_type, count):
    """
    Logs an uploaded file.
    """
    upload_log.insert_one(
        {
            "filename": filename,
            "type": dataset_type,
            "record_count": count
        }
    )

def upload_files(folder_path, collection, label):
    """
    Uploads all files present in the provided directories keeping track of the ones already uploaded.
    """
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".parquet")])

    for file in tqdm(files, desc=f"Processing {label}"):
        if already_uploaded(file, label):
            continue  # Skips file if previously uploaded

        full_path = os.path.join(folder_path, file)

        try:
            df = spark.read.parquet(full_path)
            df_clean = df.dropna().withColumn("source_file", lit(file)) # Drops NaN values and adds source file for traceability
            records = df_clean.toPandas().to_dict(orient="records")

            if records:
                collection.insert_many(records)
                log_upload(file, label, len(records))

        except Exception as e:
            print(f"[ERROR] Could not insert {file}: {e}")

if __name__ == "__main__":
    # Uploads lob and trade data to the database
    upload_files(lob_folder, lob_collection, "Landing LOB")
    upload_files(trade_folder, trade_collection, "Landing Trade")
    print("All data is up to date")
    spark.stop()