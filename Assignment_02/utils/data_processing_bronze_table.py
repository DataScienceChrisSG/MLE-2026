import os
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col


def process_bronze_table(snapshot_date_str, source_file_path, bronze_directory, table_name, spark):
    """
    Bronze layer: ingest one monthly snapshot of a raw source table AS-IS.

    No cleaning or type casting happens here - every column is read as a string so that
    the bronze table is a faithful copy of what the source system sent us. This lets us
    re-run silver/gold logic later without going back to the source system.

    Output: <bronze_directory>/bronze_<table_name>_<YYYY_MM_DD>.csv
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to source back end - IRL connect to back end source system
    # load data - IRL ingest from back end source system (only this month's snapshot)
    df = spark.read.csv(source_file_path, header=True, inferSchema=False) \
        .filter(col("snapshot_date") == snapshot_date_str)
    print(f"[bronze:{table_name}] {snapshot_date_str} row count: {df.count()}")

    # save bronze table to datamart - IRL connect to database to write
    partition_name = f"bronze_{table_name}_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = os.path.join(bronze_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    return df
