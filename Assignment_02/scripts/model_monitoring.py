"""
Monitor one model slot (or the routed production decision) for one run month.

    python scripts/model_monitoring.py --slot model_2 --snapshotdate 2024-07-01

Two checks, because labels arrive 6 months after the decision:
  - STABILITY  for applications scored this month  (available immediately)
      volume, mean predicted PD, approval rate, score PSI vs training, and for the model's top
      features: CSI vs training and missing-value rate vs training
  - PERFORMANCE for applications scored 6 months ago (their labels mature this month)
      AUC, Gini, KS, calibration (mean predicted PD vs actual default rate), precision / recall / F1,
      approval rate and the default rate among approved and declined applicants

Each metric gets a traffic-light status from the thresholds in utils/config.py.
Output (long format, one row per metric): datamart/gold/model_monitoring/<slot>/<slot>_monitoring_YYYY_MM_DD.parquet
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import enter_project_root, skip, write_parquet  # noqa: E402

enter_project_root()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import config as C  # noqa: E402
from utils import ml_data  # noqa: E402
from utils import metrics as M  # noqa: E402
from utils import model_registry as R  # noqa: E402


class Collector:
    def __init__(self, run_date, slot):
        self.rows, self.run_date, self.slot = [], run_date, slot

    def add(self, version, cohort, group, metric, value, reference=None, status="info"):
        self.rows.append({
            "monitoring_date": self.run_date, "model_slot": self.slot, "model_version": version,
            "cohort_date": cohort, "metric_group": group, "metric": metric,
            "value": float(value) if value is not None else np.nan,
            "reference_value": float(reference) if reference is not None else np.nan,
            "status": status,
        })


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------
def stability_model(col, preds, run_date):
    """model_1 / model_2: compare this month's scores and features with the training reference."""
    feats = ml_data.load_features(first=run_date, last=run_date).set_index("loan_id")
    for version, p in preds.groupby("model_version"):
        art = R.load_artefact(p["model_slot"].iloc[0], version)
        ref = art["reference"]
        col.add(version, run_date, "stability", "n_scored", len(p))
        col.add(version, run_date, "stability", "mean_score", p["score"].mean(), ref["mean_score"])
        col.add(version, run_date, "stability", "approval_rate", (p["decision"] == "approve").mean(),
                art["card"]["metrics"]["oot_approval_rate"])
        score_psi = M.psi(ref["score_props"], M.bin_proportions(p["score"].values, ref["score_edges"]))
        col.add(version, run_date, "stability", "psi_score", score_psi, None,
                M.status_high_is_bad(score_psi, C.PSI_AMBER, C.PSI_RED))
        X = feats.loc[p["loan_id"]]
        for f, r in ref["features"].items():
            csi = M.psi(r["props"], M.bin_proportions(X[f].values, r["edges"]))
            col.add(version, run_date, "stability", f"csi__{f}", csi, None, M.status_high_is_bad(csi, C.PSI_AMBER, C.PSI_RED))
            miss = float(X[f].isna().mean())
            increase = miss - r["missing_rate"]
            col.add(version, run_date, "stability", f"missing__{f}", miss, r["missing_rate"],
                    "red" if increase > C.MISSING_RATE_INCREASE_RED else "green")


def stability_production(col, preds, run_date):
    """Routed decision: volumes, routing mix and upstream data coverage (vs the go-live month)."""
    base_path = C.predictions_path(C.PRODUCTION_NAME, C.GO_LIVE_DATE)
    share_m1 = (preds["routed_to"] == "model_1").mean()
    col.add("routed", run_date, "stability", "n_scored", len(preds))
    col.add("routed", run_date, "stability", "share_routed_model_1", share_m1)
    col.add("routed", run_date, "stability", "share_routed_model_2", 1 - share_m1)
    col.add("routed", run_date, "stability", "mean_score", preds["score"].mean())
    col.add("routed", run_date, "stability", "approval_rate", (preds["decision"] == "approve").mean())
    if os.path.exists(base_path):
        base_df = pd.read_parquet(base_path)
        # data-quality check on an upstream source: share of applicants arriving with clickstream history
        coverage, base_cov = preds["has_clickstream"].mean(), base_df["has_clickstream"].mean()
        col.add("routed", run_date, "stability", "clickstream_coverage", coverage, base_cov,
                "red" if base_cov - coverage > C.MISSING_RATE_INCREASE_RED else "green")
    # score drift is measured per model against its own training reference (model_1 / model_2 tables);
    # comparing routed scores across different model versions would mix model changes with population drift


# ---------------------------------------------------------------------------
# performance (labels matured this month)
# ---------------------------------------------------------------------------
def performance(col, slot, run_date):
    cohort = C.add_months(run_date, -C.LABEL_MOB)
    path = C.predictions_path(slot, cohort)
    if cohort < C.GO_LIVE_DATE or not os.path.exists(path):
        return
    preds = pd.read_parquet(path)
    labels = ml_data.load_labels(first=run_date, last=run_date)
    df = preds.merge(labels[["loan_id", "label"]], on="loan_id", how="inner")
    if df.empty:
        return
    reg = R.load_registry()
    group_col = "model_version" if slot != C.PRODUCTION_NAME else "model_slot"
    for key, g in df.groupby(group_col):
        y, p = g["label"].astype(int).values, g["score"].values
        thr = g["threshold"].values  # per-row cut-off (production may mix models)

        if slot == C.PRODUCTION_NAME:
            # expected AUC = OOT AUC of the model(s) that made the decisions, weighted by volume
            ref_auc = sum((g["model_version"] == v).mean() * _oot_auc(reg, s, v)
                          for (s, v) in g[["routed_to", "model_version"]].drop_duplicates().itertuples(index=False))
            version = "routed"
        else:
            ref_auc, version = _oot_auc(reg, slot, key), key

        auc = M.safe_auc(y, p)
        drop = ref_auc - auc if not np.isnan(auc) else np.nan
        col.add(version, cohort, "performance", "n_labelled", len(g))
        col.add(version, cohort, "performance", "actual_default_rate", y.mean())
        col.add(version, cohort, "performance", "mean_predicted_pd", p.mean())
        gap = abs(p.mean() - y.mean())
        col.add(version, cohort, "performance", "calibration_gap", gap, None,
                M.status_high_is_bad(gap, C.CALIBRATION_GAP_AMBER, C.CALIBRATION_GAP_RED))
        col.add(version, cohort, "performance", "auc", auc, ref_auc, M.status_high_is_bad(drop, C.AUC_DROP_AMBER, C.AUC_DROP_RED))
        col.add(version, cohort, "performance", "gini", 2 * auc - 1, 2 * ref_auc - 1)
        col.add(version, cohort, "performance", "ks", M.ks_statistic(y, p))
        declined = p >= thr
        summary = M.classification_summary(y, declined.astype(float), 0.5)  # threshold already applied
        for k, v in summary.items():
            col.add(version, cohort, "performance", k, v)


def _oot_auc(reg, slot, version):
    for v in reg["slots"].get(slot, []):
        if v["version"] == version:
            return v["oot_auc"]
    return np.nan


def main(slot, run_date):
    if run_date < C.GO_LIVE_DATE:
        skip(f"monitoring starts at go-live ({C.GO_LIVE_DATE})")
    col = Collector(run_date, slot)

    path = C.predictions_path(slot, run_date)
    if os.path.exists(path):
        preds = pd.read_parquet(path)
        if slot == C.PRODUCTION_NAME:
            stability_production(col, preds, run_date)
        else:
            stability_model(col, preds, run_date)
    performance(col, slot, run_date)

    if not col.rows:
        skip(f"[{slot}] nothing to monitor on {run_date}: no applications scored this month "
             f"and no scored cohort whose labels matured this month")
    out = pd.DataFrame(col.rows)
    reds = out[out["status"] == "red"]
    print(f"[{slot}] {len(out)} metrics | red alerts: {len(reds)}")
    for _, r in reds.iterrows():
        print(f"   RED {r['metric_group']}/{r['metric']} = {r['value']:.3f} (cohort {r['cohort_date']})")
    write_parquet(out, C.monitoring_path(slot, run_date))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slot", required=True, choices=list(C.MODEL_SLOTS) + [C.PRODUCTION_NAME])
    p.add_argument("--snapshotdate", required=True)
    a = p.parse_args()
    main(a.slot, a.snapshotdate)
