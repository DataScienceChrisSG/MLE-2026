"""
Render the monitoring charts and HTML report from the gold monitoring tables and the model registry.

    python scripts/visualise_monitoring.py --snapshotdate 2025-07-01

Output: reports/monitoring/*.png and reports/monitoring_report.html (open in any browser).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import enter_project_root, skip, read_parquet_folder  # noqa: E402

enter_project_root()

import pandas as pd  # noqa: E402

from utils import config as C  # noqa: E402
from utils import model_registry as R  # noqa: E402
from utils import visualisation as V  # noqa: E402


def main(run_date):
    frames = [read_parquet_folder(os.path.join(C.DIRS["gold"]["model_monitoring"], s))
              for s in list(C.MODEL_SLOTS) + [C.PRODUCTION_NAME]]
    mon = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if mon.empty:
        skip("no monitoring results yet (monitoring starts at go-live)")
    mon = mon[mon["monitoring_date"] <= run_date]   # report as of this run (idempotent re-runs)
    reg = R.load_registry()

    out_dir = os.path.join(C.REPORTS_DIR, "monitoring")
    os.makedirs(out_dir, exist_ok=True)
    outage = V.outage_start(mon)
    charts = [
        ("Model performance", V.chart_performance(mon, reg, out_dir, outage)),
        ("Score stability", V.chart_stability(mon, out_dir, outage)),
        ("Calibration", V.chart_calibration(mon, out_dir, outage)),
        ("Routing and volume", V.chart_routing(mon, out_dir)),
        ("Business outcomes", V.chart_business(mon, out_dir, outage)),
        ("Feature drift", V.chart_feature_drift(mon, out_dir)),
        ("Model versions", V.chart_model_timeline(reg, out_dir)),
    ]
    report = V.build_report(mon, reg, charts, os.path.join(C.REPORTS_DIR, "monitoring_report.html"), run_date)
    print(f"charts: {[os.path.basename(p) for _, p in charts if p]}")
    print("report:", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    main(p.parse_args().snapshotdate)
