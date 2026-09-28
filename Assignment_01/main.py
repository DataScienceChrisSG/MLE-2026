import os
import glob
import argparse
from datetime import datetime

import pyspark
import pyspark.sql.functions as F

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# ---------------------------------------------------------------------------
# run mode
#   python main.py                               -> full backfill (all months)
#   python main.py --snapshotdate 2024-06-01     -> one month only, as a scheduled monthly job would
#                                                   (assumes earlier months already exist in the datamart)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Bronze -> silver -> gold pipeline for the loan default feature and label stores")
parser.add_argument("--snapshotdate", type=str, default=None, help="YYYY-MM-DD (first of month). Omit to backfill all months.")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
# Loan applications (and therefore feature snapshots) run from Jan 2023 to Jan 2025.
feature_start_date_str = "2023-01-01"
feature_end_date_str = "2025-01-01"

# Loan repayment data (LMS) runs Jan 2023 - Nov 2025: the last loans start Jan 2025 and mature
# after 10 instalments in Nov 2025. Bronze/silver hold the full source history.
lms_start_date_str = "2023-01-01"
lms_end_date_str = "2025-11-01"

# Labels (dpd at mob 6) exist up to Jul 2025 (Jan 2025 applications + 6 months);
# later months have no loan at mob 6, so they would only produce empty label partitions.
label_start_date_str = "2023-01-01"
label_end_date_str = "2025-07-01"

LABEL_DPD = 30
LABEL_MOB = 6

# raw source files (treated as the source systems)
SOURCES = {
    "loan_daily": "data/lms_loan_daily.csv",
    "attributes": "data/features_attributes.csv",
    "financials": "data/features_financials.csv",
    "clickstream": "data/feature_clickstream.csv",
}

# datamart layout
DIRS = {
    "bronze": {
        "loan_daily": "datamart/bronze/lms/",
        "attributes": "datamart/bronze/attributes/",
        "financials": "datamart/bronze/financials/",
        "clickstream": "datamart/bronze/clickstream/",
    },
    "silver": {
        "loan_daily": "datamart/silver/loan_daily/",
        "attributes": "datamart/silver/attributes/",
        "financials": "datamart/silver/financials/",
        "clickstream": "datamart/silver/clickstream/",
    },
    "gold": {
        "label_store": "datamart/gold/label_store/",
        "feature_store": "datamart/gold/feature_store/",
    },
}


# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)
    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    return first_of_month_dates


feature_dates = generate_first_of_month_dates(feature_start_date_str, feature_end_date_str)
lms_dates = generate_first_of_month_dates(lms_start_date_str, lms_end_date_str)
label_dates = generate_first_of_month_dates(label_start_date_str, label_end_date_str)
clickstream_dates = feature_dates

if args.snapshotdate:
    run_date = args.snapshotdate
    # clickstream features average ALL earlier months, which are already in silver from previous runs
    clickstream_dates = [d for d in clickstream_dates if d == run_date]
    feature_dates = [d for d in feature_dates if d == run_date]
    lms_dates = [d for d in lms_dates if d == run_date]
    label_dates = [d for d in label_dates if d == run_date]
    print(f"single-month run for {run_date}")
print("feature snapshot dates:", feature_dates)
print("loan (LMS) snapshot dates:", lms_dates)
print("label snapshot dates:", label_dates)

for layer in DIRS.values():
    for d in layer.values():
        os.makedirs(d, exist_ok=True)

# which dates each source table is processed for
TABLE_DATES = {
    "loan_daily": lms_dates,
    "attributes": feature_dates,
    "financials": feature_dates,
    "clickstream": clickstream_dates,
}


# ---------------------------------------------------------------------------
# BRONZE: raw monthly snapshots, as-is
# ---------------------------------------------------------------------------
print("\n========== BRONZE ==========")
for table_name, dates in TABLE_DATES.items():
    for date_str in dates:
        utils.data_processing_bronze_table.process_bronze_table(
            date_str, SOURCES[table_name], DIRS["bronze"][table_name], table_name, spark)


# ---------------------------------------------------------------------------
# SILVER: cleaned, typed, validated
# ---------------------------------------------------------------------------
print("\n========== SILVER ==========")
silver_jobs = {
    "loan_daily": utils.data_processing_silver_table.process_silver_loan_daily,
    "attributes": utils.data_processing_silver_table.process_silver_attributes,
    "financials": utils.data_processing_silver_table.process_silver_financials,
    "clickstream": utils.data_processing_silver_table.process_silver_clickstream,
}
for table_name, job in silver_jobs.items():
    for date_str in TABLE_DATES[table_name]:
        job(date_str, DIRS["bronze"][table_name], DIRS["silver"][table_name], spark)

utils.data_processing_silver_table.check_silver_attributes_quality(DIRS["silver"]["attributes"], spark)


# ---------------------------------------------------------------------------
# GOLD: ML-ready label store and feature store
# ---------------------------------------------------------------------------
print("\n========== GOLD ==========")
for date_str in label_dates:
    utils.data_processing_gold_table.process_labels_gold_table(
        date_str, DIRS["silver"]["loan_daily"], DIRS["gold"]["label_store"], spark, dpd=LABEL_DPD, mob=LABEL_MOB)

for date_str in feature_dates:
    utils.data_processing_gold_table.process_features_gold_table(
        date_str,
        DIRS["silver"]["loan_daily"],
        DIRS["silver"]["attributes"],
        DIRS["silver"]["financials"],
        DIRS["silver"]["clickstream"],
        DIRS["gold"]["feature_store"],
        spark,
    )


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def read_gold(folder_path):
    files_list = glob.glob(os.path.join(folder_path, "*.parquet"))
    return spark.read.parquet(*files_list)

labels = read_gold(DIRS["gold"]["label_store"])
features = read_gold(DIRS["gold"]["feature_store"])
print("\nlabel_store row_count:", labels.count())
print("feature_store row_count:", features.count(), "| columns:", len(features.columns))

# features and labels join on loan_id; the label is observed LABEL_MOB months after application
joined = features.alias("f").join(labels.alias("l"), F.col("f.loan_id") == F.col("l.loan_id"), "inner")
print("feature rows with a matured label:", joined.count())
misaligned = joined.filter(F.col("l.snapshot_date") != F.add_months(F.col("f.snapshot_date"), LABEL_MOB)).count()
print(f"integrity check - labels not observed exactly {LABEL_MOB} months after application:", misaligned)
labels.groupBy("label").count().show()

spark.stop()
