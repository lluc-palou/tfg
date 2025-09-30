# =================================================================================================
# Imports                                                                                        
# =================================================================================================

import os
import sys
import time
import json
import warnings
import numpy as np
from scipy import stats
from datetime import datetime
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, date_trunc, expr, lit

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
FEATURES_LOB_COLLECTION = "features_lob_train"
TRANSFORMED_LOB_COLLECTION = "transformed_lob_train"
METADATA_DIR = "C:/Users/llucp/desktop/tfg_metadata"

spark = (
    SparkSession.builder
    .appName("Stage5_FeatureTransformation")
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

# =================================================================================================
# Utilities                                                                                    
# =================================================================================================

def logger(msg: str) -> None:
    """
    Shows workflow messages while executing script through terminal.
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[LOB FEATURE TRANSFORMATION] [{ts}] {msg}")
    sys.stdout.flush()

def load_features_sample(sample_size: int = 10000) -> DataFrame:
    """
    Loads a sample of non-context features for transformation fitting.
    """
    logger(f'Loading {sample_size} non-context samples for transformation fitting')
    
    pipeline = [
        {"$match": {"is_context": False}},
        {"$sample": {"size": sample_size}},
        {"$project": {"_id": 0}}
    ]
    
    pipeline_json = json.dumps(pipeline)
    
    features_sample = (
        spark.read.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", FEATURES_LOB_COLLECTION)
        .option("aggregation.pipeline", pipeline_json)
        .load()
    )
    
    count = features_sample.count()
    logger(f'Loaded {count} non-context sample records for fitting')
    
    return features_sample

def load_all_features() -> DataFrame:
    """
    Loads all features including context rows, sorted by timestamp.
    """
    logger(f'Loading all features from {FEATURES_LOB_COLLECTION}')
    
    pipeline = [
        {"$sort": {"timestamp": 1}},
        {"$project": {"_id": 0}}
    ]
    
    pipeline_json = json.dumps(pipeline)
    
    features_df = (
        spark.read.format("mongodb")
        .option("database", DB_NAME)
        .option("collection", FEATURES_LOB_COLLECTION)
        .option("aggregation.pipeline", pipeline_json)
        .load()
    )
    
    total_count = features_df.count()
    
    if 'is_context' in features_df.columns:
        context_count = features_df.filter(col('is_context') == True).count()
        non_context_count = features_df.filter(col('is_context') == False).count()
        logger(f'Loaded {total_count} total records: {context_count} context + {non_context_count} non-context')
    else:
        logger(f'Loaded {total_count} feature records')
    
    return features_df

def load_transformation_metadata() -> dict:
    """
    Loads transformation metadata from JSON file.
    """
    logger(f'Loading transformation metadata from {METADATA_DIR}')
    
    input_path = os.path.join(METADATA_DIR, 'transformation_metadata.json')
    
    if not os.path.exists(input_path):
        logger(f'ERROR: No transformation metadata found at {input_path}')
        return None
    
    with open(input_path, 'r') as f:
        metadata_doc = json.load(f)
    
    logger(f'Loaded metadata with {metadata_doc["num_features"]} transformations')
    
    return metadata_doc['transformations']

def save_transformation_metadata(transformation_map: dict) -> None:
    """
    Saves transformation metadata to JSON file.
    """
    logger(f'Saving transformation metadata to {METADATA_DIR}')
    
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    metadata_doc = {
        'created_at': datetime.now().isoformat(),
        'source_collection': FEATURES_LOB_COLLECTION,
        'target_collection': TRANSFORMED_LOB_COLLECTION,
        'sample_size': 10000,
        'num_features': len(transformation_map),
        'context_included': True,
        'note': 'Transformations fitted on non-context data but applied to all data',
        'transformations': transformation_map
    }
    
    output_path = os.path.join(METADATA_DIR, 'transformation_metadata.json')
    with open(output_path, 'w') as f:
        json.dump(metadata_doc, f, indent=2)
    
    logger(f'Transformation metadata saved to {output_path}')

def write_transformed_features(df: DataFrame) -> None:
    """
    Writes transformed features to the provided database collection.
    """
    (df.write.format("mongodb")
       .option("database", DB_NAME)
       .option("collection", TRANSFORMED_LOB_COLLECTION)
       .option("ordered", "false")
       .mode("append")
       .save())

# =================================================================================================
# Feature Selection                                                                          
# =================================================================================================

# Columns to exclude from transformation
EXCLUDE_COLUMNS = [
    'timestamp',
    'trading_hour',
    'bids',
    'asks',
    'mid_price',      # Used to standardize prices in raw LOB standardization
    'variance_proxy', # Used to standardize prices in raw LOB standardization
    'is_context',
]

# Prefixes to exclude (targets, not features)
EXCLUDE_PREFIXES = [
    'fwd_logret_',
]

def select_features_to_transform(df: DataFrame) -> list:
    """
    Dynamically identify features to transform by excluding identifiers, 
    raw LOB data, context flags, and target variables.
    """
    all_columns = df.columns
    features_to_transform = []
    
    for col_name in all_columns:
        if col_name in EXCLUDE_COLUMNS:
            continue
        
        if any(col_name.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            continue
        
        features_to_transform.append(col_name)
    
    logger(f'Identified {len(features_to_transform)} features to transform')
    logger(f'Excluded {len(all_columns) - len(features_to_transform)} columns')
    
    return features_to_transform

# =================================================================================================
# Normality Metrics                                                                               
# =================================================================================================

def calculate_normality_score(data: np.ndarray) -> dict:
    """
    Calculates normality metrics for a data array, returning individual 
    metrics and a combined score.
    """
    data = data[np.isfinite(data)]
    
    if len(data) < 3:
        return {
            'shapiro_pvalue': 0.0,
            'skewness': 999.0,
            'excess_kurtosis': 999.0,
            'combined_score': -999.0
        }
    
    # Shapiro-Wilk test (sample if too large)
    if len(data) > 5000:
        sample_idx = np.random.choice(len(data), 5000, replace=False)
        shapiro_stat, shapiro_pval = stats.shapiro(data[sample_idx])
    else:
        shapiro_stat, shapiro_pval = stats.shapiro(data)
    
    # Moments
    skewness = float(stats.skew(data))
    kurtosis = float(stats.kurtosis(data))
    
    # Combined score: prioritize Shapiro, penalize skewness and kurtosis
    combined_score = shapiro_pval - 0.1 * abs(skewness) - 0.05 * abs(kurtosis)
    
    return {
        'shapiro_pvalue': float(shapiro_pval),
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'combined_score': combined_score
    }

# =================================================================================================
# Transformation Fitting                                                                          
# =================================================================================================

def fit_best_transformation(feature_data: np.ndarray, feature_name: str) -> dict:
    """
    Tests multiple transformations and selects the best based on normality.
    Returns dict with transformation metadata.
    """
    logger(f'Fitting transformations for {feature_name}')
    
    valid_data = feature_data[np.isfinite(feature_data)]
    
    if len(valid_data) < 10:
        logger(f'  {feature_name}: insufficient data, using identity')
        return {
            'transformation': 'identity',
            'parameters': {},
            'normality_score': -999.0,
            'original_stats': {},
            'transformed_stats': {}
        }
    
    # Calculate original statistics
    try:
        orig_stats = {
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'skew': float(stats.skew(valid_data))
        }
    except Exception as e:
        logger(f'  {feature_name}: failed to compute stats - {e}, using identity')
        return {
            'transformation': 'identity',
            'parameters': {},
            'normality_score': -999.0,
            'original_stats': {},
            'transformed_stats': {}
        }
    
    # Define transformation candidates
    candidates = []
    
    # Identity (baseline)
    candidates.append(('identity', {}, valid_data.copy()))
    
    # Asinh (works for all values)
    try:
        transformed = np.arcsinh(valid_data)
        if np.all(np.isfinite(transformed)):
            candidates.append(('asinh', {}, transformed))
    except Exception as e:
        logger(f'  {feature_name}: asinh failed - {e}')
    
    # Yeo-Johnson (works for all values)
    try:
        with np.errstate(all='ignore'):
            transformed_yj, lmbda_yj = stats.yeojohnson(valid_data)
        if np.all(np.isfinite(transformed_yj)):
            candidates.append(('yeo-johnson', {'lambda': float(lmbda_yj)}, transformed_yj))
    except Exception as e:
        logger(f'  {feature_name}: yeo-johnson failed - {e}')
    
    # Log (if all positive or can shift)
    min_val = np.min(valid_data)
    try:
        if min_val > 0:
            transformed = np.log(valid_data)
            if np.all(np.isfinite(transformed)):
                candidates.append(('log', {'offset': 0.0}, transformed))
        elif min_val >= -1e6:
            offset = abs(min_val) + 1.0
            transformed = np.log(valid_data + offset)
            if np.all(np.isfinite(transformed)):
                candidates.append(('log', {'offset': float(offset)}, transformed))
    except Exception as e:
        logger(f'  {feature_name}: log failed - {e}')
    
    # Sqrt (if all positive or can shift)
    try:
        if min_val >= 0:
            transformed = np.sqrt(valid_data)
            if np.all(np.isfinite(transformed)):
                candidates.append(('sqrt', {'offset': 0.0}, transformed))
        elif min_val >= -1e6:
            offset = abs(min_val) + 0.1
            transformed = np.sqrt(valid_data + offset)
            if np.all(np.isfinite(transformed)):
                candidates.append(('sqrt', {'offset': float(offset)}, transformed))
    except Exception as e:
        logger(f'  {feature_name}: sqrt failed - {e}')
    
    # Box-Cox (only if all positive)
    if min_val > 0:
        try:
            with np.errstate(all='ignore'):
                transformed_bc, lmbda_bc = stats.boxcox(valid_data)
            if np.all(np.isfinite(transformed_bc)):
                candidates.append(('box-cox', {'lambda': float(lmbda_bc), 'offset': 0.0}, transformed_bc))
        except Exception as e:
            logger(f'  {feature_name}: box-cox failed - {e}')
    
    # Evaluate each candidate
    best_score = -np.inf
    best_transform = None
    
    for name, params, transformed in candidates:
        try:
            score_dict = calculate_normality_score(transformed)
            score = score_dict['combined_score']
            
            if score > best_score:
                best_score = score
                best_transform = {
                    'transformation': name,
                    'parameters': params,
                    'normality_score': score,
                    'normality_metrics': score_dict,
                    'original_stats': orig_stats,
                    'transformed_stats': {
                        'mean': float(np.mean(transformed[np.isfinite(transformed)])),
                        'std': float(np.std(transformed[np.isfinite(transformed)])),
                        'skew': score_dict['skewness']
                    }
                }
        except Exception as e:
            logger(f'  {feature_name}: failed to score {name} - {e}')
            continue
    
    if best_transform is None:
        logger(f'  {feature_name}: all transformations failed, using identity')
        best_transform = {
            'transformation': 'identity',
            'parameters': {},
            'normality_score': -999.0,
            'normality_metrics': {
                'shapiro_pvalue': 0.0,
                'skewness': 999.0,
                'excess_kurtosis': 999.0,
                'combined_score': -999.0
            },
            'original_stats': orig_stats,
            'transformed_stats': orig_stats
        }
    
    logger(f'  {feature_name}: selected {best_transform["transformation"]} (score={best_transform["normality_score"]:.4f})')
    
    return best_transform

def fit_all_transformations(features_df: DataFrame) -> dict:
    """
    Fits transformations for all LOB-derived features, returning a transformation map.
    """
    logger('Fitting feature transformations')
    
    features_to_transform = select_features_to_transform(features_df)
    
    if not features_to_transform:
        logger('ERROR: No features found to transform')
        return {}
    
    logger(f'Features to transform: {features_to_transform}')
    
    transformation_map = {}
    
    logger('Converting sample to Pandas for transformation fitting')
    pandas_df = features_df.select(features_to_transform).toPandas()
    
    for feature_name in features_to_transform:
        feature_data = pandas_df[feature_name].values
        transform_info = fit_best_transformation(feature_data, feature_name)
        transformation_map[feature_name] = transform_info
    
    logger(f'Transformation fitting complete - fitted {len(transformation_map)} transformations')
    
    return transformation_map

# =================================================================================================
# Transformation Application                                                                      
# =================================================================================================

def get_all_hours(df: DataFrame) -> list:
    """
    Extracts all trading hours present in the dataset.
    """
    logger('Extracting all trading hours')
    
    all_hours = (df.withColumn("trading_hour", date_trunc("hour", col("timestamp")))
                   .select("trading_hour")
                   .distinct()
                   .orderBy("trading_hour")
                   .collect())
    
    available_hours = [row.trading_hour for row in all_hours]
    logger(f'Found {len(available_hours)} hours from {available_hours[0]} to {available_hours[-1]}')
    
    return available_hours

def load_hour_batch(df: DataFrame, target_hour) -> DataFrame:
    """
    Loads all data for a specific trading hour.
    """
    hour_start = target_hour
    hour_end = expr(f"'{target_hour}' + INTERVAL 1 HOURS")
    
    filter_condition = (
        (col("timestamp") >= lit(hour_start)) &
        (col("timestamp") < hour_end)
    )
    
    logger(f'Loading hour {target_hour}')
    
    return df.filter(filter_condition)

def apply_transformations_to_hour(hour_df: DataFrame, transformation_map: dict) -> DataFrame:
    """
    Applies fitted transformations to one hour's worth of features.
    Handles missing values by preserving them as None/null.
    """
    result_df = hour_df
    
    for feature_name, transform_info in transformation_map.items():
        if feature_name not in hour_df.columns:
            continue
        
        transform_name = transform_info['transformation']
        params = transform_info['parameters']
        
        if transform_name == 'identity':
            continue
        
        elif transform_name == 'log':
            offset = params.get('offset', 0.0)
            def log_udf(x):
                if x is None or not np.isfinite(x):
                    return None
                val = x + offset
                if val <= 0 or not np.isfinite(val):
                    return None
                return float(np.log(val))
            transform_udf = udf(log_udf, DoubleType())
        
        elif transform_name == 'sqrt':
            offset = params.get('offset', 0.0)
            def sqrt_udf(x):
                if x is None or not np.isfinite(x):
                    return None
                val = x + offset
                if val < 0 or not np.isfinite(val):
                    return None
                return float(np.sqrt(val))
            transform_udf = udf(sqrt_udf, DoubleType())
        
        elif transform_name == 'asinh':
            def asinh_udf(x):
                if x is None or not np.isfinite(x):
                    return None
                return float(np.arcsinh(x))
            transform_udf = udf(asinh_udf, DoubleType())
        
        elif transform_name == 'yeo-johnson':
            lmbda = params.get('lambda', 1.0)
            def yj_transform(x):
                if x is None or not np.isfinite(x):
                    return None
                try:
                    result = stats.yeojohnson([x], lmbda=lmbda)[0]
                    if np.isfinite(result):
                        return float(result)
                    return None
                except:
                    return None
            transform_udf = udf(yj_transform, DoubleType())
        
        elif transform_name == 'box-cox':
            lmbda = params.get('lambda', 1.0)
            offset = params.get('offset', 0.0)
            def bc_transform(x):
                if x is None or not np.isfinite(x):
                    return None
                val = x + offset
                if val <= 0 or not np.isfinite(val):
                    return None
                try:
                    result = stats.boxcox([val], lmbda=lmbda)[0]
                    if np.isfinite(result):
                        return float(result)
                    return None
                except:
                    return None
            transform_udf = udf(bc_transform, DoubleType())
        
        else:
            continue
        
        result_df = result_df.withColumn(feature_name, transform_udf(col(feature_name)))
    
    return result_df

# =================================================================================================
# Pipeline                                                                                        
# =================================================================================================

def process_hourly_batches(features_df: DataFrame, transformation_map: dict) -> None:
    """
    Processes transformations hour by hour.
    """
    logger('Applying transformations in hourly batches')
    
    all_hours = get_all_hours(features_df)
    
    logger(f'Processing {len(all_hours)} hours')
    
    total_processed = 0
    total_time = 0
    
    for i, target_hour in enumerate(all_hours):
        batch_start_time = time.time()
        logger(f'Processing {i+1}/{len(all_hours)} - HOUR: {target_hour}')
        
        hour_batch = load_hour_batch(features_df, target_hour)
        batch_count = hour_batch.count()
        logger(f'Loaded {batch_count} rows for {target_hour}')
        
        if batch_count > 0:
            logger(f'Applying transformations')
            transformed_hour = apply_transformations_to_hour(hour_batch, transformation_map)
            
            logger(f'Writing to MongoDB')
            transformed_coalesced = transformed_hour.coalesce(1)
            write_transformed_features(transformed_coalesced)
            
            processed_count = transformed_hour.count()
            total_processed += processed_count
            logger(f'Wrote {processed_count} rows')
            
            transformed_hour.unpersist()
        else:
            logger(f'Skipping empty hour {target_hour}')
        
        batch_duration = time.time() - batch_start_time
        total_time += batch_duration
        avg_time = total_time / (i + 1)
        eta = avg_time * (len(all_hours) - i - 1)
        
        logger(f'Batch completed in {batch_duration:.2f}s, ETA: {eta:.2f}s')
        
        hour_batch.unpersist()
    
    logger(f'Transformation completed - total: {total_processed} rows in {total_time:.2f}s')

def preprocess_stage5_transformation(fit_on_sample: bool = True, sample_size: int = 10000) -> None:
    """
    Main transformation pipeline with hourly batch processing.
    Fits transformations on non-context sample, applies to all data.
    """
    logger('STAGE 5 LOB FEATURE TRANSFORMATION: START')
    
    if fit_on_sample:
        logger('Fitting transformations on non-context sample')
        features_sample = load_features_sample(sample_size)
        transformation_map = fit_all_transformations(features_sample)
        save_transformation_metadata(transformation_map)
    else:
        logger('Loading existing transformation metadata')
        transformation_map = load_transformation_metadata()
        if transformation_map is None:
            logger('ERROR: No transformations found. Run with fit_on_sample=True first')
            return
    
    logger('Loading all features including context rows')
    all_features = load_all_features()
    
    logger('Applying transformations hour by hour')
    process_hourly_batches(all_features, transformation_map)
    
    logger('STAGE 5 COMPLETED')

# =================================================================================================
# Main                                                                                       
# =================================================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    logger('STAGE 5: LOB FEATURE TRANSFORMATION PIPELINE')
    logger(f'Source database collection: {FEATURES_LOB_COLLECTION}')
    logger(f'Target database collection: {TRANSFORMED_LOB_COLLECTION}')
    logger(f'Metadata directory: {METADATA_DIR}')
    
    logger('Fitting and applying feature transformations')
    preprocess_stage5_transformation(fit_on_sample=True, sample_size=10000)
    
    total_time = time.time() - start_time
    logger(f'STAGE 5 completed in {total_time:.2f} seconds')
    
    spark.stop()