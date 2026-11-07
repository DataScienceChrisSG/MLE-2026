"""
Dependency check before the monthly run: every source file exists and has the columns the pipeline expects.
Months outside the pipeline window exit with 99 (skipped), so a manual trigger "today" does nothing harmful.

    python scripts/check_source_data.py --snapshotdate 2024-01-01
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import enter_project_root, skip  # noqa: E402

enter_project_root()

from utils import config as C  # noqa: E402

REQUIRED_COLUMNS = {
    "loan_daily": {"loan_id", "Customer_ID", "loan_start_date", "installment_num", "due_amt", "overdue_amt", "snapshot_date"},
    "attributes": {"Customer_ID", "Age", "Occupation", "snapshot_date"},
    "financials": {"Customer_ID", "Annual_Income", "Outstanding_Debt", "Credit_Mix", "snapshot_date"},
    "clickstream": {"Customer_ID", "fe_1", "fe_20", "snapshot_date"},
}


def main(run_date):
    if not C.in_range(run_date, C.PIPELINE_START, C.PIPELINE_END):
        skip(f"{run_date} is outside the pipeline window {C.PIPELINE_START} to {C.PIPELINE_END}")
    for table, path in C.SOURCES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"source file missing: {path}")
        with open(path, newline="") as f:
            header = set(next(csv.reader(f)))
        missing = REQUIRED_COLUMNS[table] - header
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        print(f"OK {table}: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    main(p.parse_args().snapshotdate)
