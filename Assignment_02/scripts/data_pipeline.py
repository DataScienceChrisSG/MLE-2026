"""
Medallion data pipeline - one table, one layer, one month per call (one Airflow task).

    python scripts/data_pipeline.py --layer bronze --table clickstream --snapshotdate 2024-01-01
    python scripts/data_pipeline.py --layer gold   --table feature_store --snapshotdate 2024-01-01

The processing logic is the Assignment 1 code in utils/data_processing_*_table.py; the only change is that
silver and gold parquet files are written safely (temporary folder, verified, then swapped in).
Months outside a table's data range exit with code 99, which Airflow shows as "skipped".
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import enter_project_root, skip, get_spark  # noqa: E402

enter_project_root()

from utils import config as C  # noqa: E402
import utils.data_processing_bronze_table as bronze  # noqa: E402
import utils.data_processing_silver_table as silver  # noqa: E402
import utils.data_processing_gold_table as gold  # noqa: E402


SILVER_JOBS = {
    "loan_daily": silver.process_silver_loan_daily,
    "attributes": silver.process_silver_attributes,
    "financials": silver.process_silver_financials,
    "clickstream": silver.process_silver_clickstream,
}
VALID = {
    "bronze": list(C.SOURCES),
    "silver": list(C.SOURCES),
    "gold": ["label_store", "feature_store"],
}


def main(layer, table, snapshot_date):
    if table not in VALID[layer]:
        raise ValueError(f"{table} is not a {layer} table; choose from {VALID[layer]}")

    start, end = C.TABLE_DATE_RANGES[table]
    if not C.in_range(snapshot_date, start, end):
        skip(f"{layer}/{table}: no source data for {snapshot_date} (available {start} to {end})")

    for d in C.DIRS[layer].values():
        os.makedirs(d, exist_ok=True)

    spark = get_spark(f"{layer}_{table}")
    try:
        if layer == "bronze":
            bronze.process_bronze_table(snapshot_date, C.SOURCES[table], C.DIRS["bronze"][table], table, spark)

        elif layer == "silver":
            SILVER_JOBS[table](snapshot_date, C.DIRS["bronze"][table], C.DIRS["silver"][table], spark)
            if table == "attributes":
                silver.check_silver_attributes_quality(C.DIRS["silver"]["attributes"], spark)

        elif table == "label_store":
            gold.process_labels_gold_table(snapshot_date, C.DIRS["silver"]["loan_daily"],
                                           C.DIRS["gold"]["label_store"], spark, dpd=C.LABEL_DPD, mob=C.LABEL_MOB)
        else:
            gold.process_features_gold_table(
                snapshot_date,
                C.DIRS["silver"]["loan_daily"], C.DIRS["silver"]["attributes"],
                C.DIRS["silver"]["financials"], C.DIRS["silver"]["clickstream"],
                C.DIRS["gold"]["feature_store"], spark,
            )
    finally:
        spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run one medallion table for one month")
    p.add_argument("--layer", required=True, choices=["bronze", "silver", "gold"])
    p.add_argument("--table", required=True)
    p.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD (first of month)")
    a = p.parse_args()
    main(a.layer, a.table, a.snapshotdate)
