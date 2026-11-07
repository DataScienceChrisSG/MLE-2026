"""
Score this month's loan applications with the model that is live for a slot, or route the final decision.

    python scripts/model_inference.py --slot model_1 --snapshotdate 2024-03-01
    python scripts/model_inference.py --slot model_2 --snapshotdate 2024-03-01
    python scripts/model_inference.py --slot production --snapshotdate 2024-03-01

model_1 scores applicants WITH clickstream history, model_2 scores ALL applicants (it is the production
model for applicants without clickstream and a shadow benchmark for the rest). "production" routes:
model_1's score where the applicant has clickstream, otherwise model_2's.

Only applications made in the run month are scored (no in-sample scoring of training months).
Output: datamart/gold/model_predictions/<slot>/<slot>_predictions_YYYY_MM_DD.parquet
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import enter_project_root, skip, write_parquet  # noqa: E402

enter_project_root()

import pandas as pd  # noqa: E402

from utils import config as C  # noqa: E402
from utils import ml_data  # noqa: E402
from utils import model_registry as R  # noqa: E402


def score_slot(slot, run_date):
    apps = ml_data.load_features(first=run_date, last=run_date)
    if apps.empty:
        skip(f"no loan applications in {run_date}")
    apps = ml_data.slot_population(apps, slot)
    if apps.empty:
        skip(f"[{slot}] no applicants with clickstream history in {run_date}: every applicant is routed to model_2")

    reg = R.load_registry()
    entry = R.champion_at(reg, slot, run_date)
    if entry is None:
        raise RuntimeError(f"no live {slot} model on {run_date} - check the model registry")
    art = R.load_artefact(slot, entry["version"])

    # schema check: the feature store must still provide every feature the model was trained on
    missing = [f for f in art["features"] if f not in apps.columns]
    if missing:
        raise RuntimeError(f"feature store is missing {len(missing)} model features: {missing[:5]}")

    scores = art["model"].predict_proba(apps[art["features"]].astype(float))[:, 1]
    out = apps[["loan_id", "Customer_ID", "snapshot_date", "has_clickstream"]].copy()
    out["model_slot"] = slot
    out["model_version"] = entry["version"]
    out["algorithm"] = entry["algorithm"]
    out["score"] = scores                                    # predicted probability of default
    out["threshold"] = art["threshold"]
    out["decision"] = (out["score"] >= art["threshold"]).map({True: "decline", False: "approve"})
    out["scored_on"] = run_date
    print(f"[{slot}] scored {len(out)} applications with {entry['version']} ({entry['algorithm']}); "
          f"mean PD {scores.mean():.3f}, approval rate {(out['decision'] == 'approve').mean():.2f}")
    write_parquet(out, C.predictions_path(slot, run_date))


def route(run_date):
    parts = []
    for slot in C.MODEL_SLOTS:
        p = C.predictions_path(slot, run_date)
        if os.path.exists(p):
            parts.append(pd.read_parquet(p))
    if not parts:
        skip(f"no model predictions for {run_date}")
    preds = pd.concat(parts, ignore_index=True)
    use_model_1 = (preds["model_slot"] == "model_1") | (
        (preds["model_slot"] == "model_2") & ~preds["loan_id"].isin(preds.loc[preds["model_slot"] == "model_1", "loan_id"]))
    routed = preds[use_model_1].copy()
    routed = routed.rename(columns={"model_slot": "routed_to"})
    routed.insert(3, "model_slot", C.PRODUCTION_NAME)
    assert routed["loan_id"].is_unique
    share = (routed["routed_to"] == "model_1").mean()
    print(f"[production] {len(routed)} decisions: {share:.0%} by model_1 (clickstream), {1-share:.0%} by model_2")
    write_parquet(routed, C.predictions_path(C.PRODUCTION_NAME, run_date))


def main(slot, run_date):
    if run_date < C.GO_LIVE_DATE:
        skip(f"model goes live on {C.GO_LIVE_DATE}; nothing is scored before that")
    if slot == C.PRODUCTION_NAME:
        route(run_date)
    else:
        score_slot(slot, run_date)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slot", required=True, choices=list(C.MODEL_SLOTS) + [C.PRODUCTION_NAME])
    p.add_argument("--snapshotdate", required=True)
    a = p.parse_args()
    main(a.slot, a.snapshotdate)
