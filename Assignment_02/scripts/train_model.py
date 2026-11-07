"""
Retrain one model slot if the retraining policy says so (one Airflow task per slot).

    python scripts/train_model.py --slot model_1 --snapshotdate 2024-03-01

Policy (utils/config.py):
  - first deployment at GO_LIVE_DATE
  - scheduled retrain every RETRAIN_EVERY_MONTHS months after go-live
  - off-cycle retrain if monitoring raised a red alert for this slot since its last training
  - skip if no new labelled data exists since the last training (nothing new to learn from)
The new model is saved as a CANDIDATE; evaluate_and_promote.py decides whether it goes live.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import enter_project_root, skip, read_parquet_folder  # noqa: E402

enter_project_root()

from utils import config as C  # noqa: E402
from utils import ml_data  # noqa: E402
from utils import model_registry as R  # noqa: E402
from utils import model_training  # noqa: E402

TRIGGER_METRICS = {"auc", "calibration_gap", "psi_score"}


def red_alerts_since(slot, since, before):
    mon = read_parquet_folder(os.path.join(C.DIRS["gold"]["model_monitoring"], slot))
    if mon.empty:
        return mon
    mon = mon[(mon["monitoring_date"] > since) & (mon["monitoring_date"] < before)]
    return mon[(mon["status"] == "red") & (mon["metric"].isin(TRIGGER_METRICS))]


def retrain_decision(slot, run_date, reg):
    if run_date < C.GO_LIVE_DATE:
        return False, f"before go-live ({C.GO_LIVE_DATE})"
    if run_date > C.FEATURE_END:
        return False, f"no new applications to score after {C.FEATURE_END}, so a new model would never be used"

    trained = [v for v in reg["slots"].get(slot, []) if v["trained_on"] < run_date]
    if not trained:
        return True, "first deployment (go-live)"
    last = max(trained, key=lambda v: v["trained_on"])

    # is there labelled data we have not trained on yet?
    data = ml_data.slot_population(ml_data.labelled_data(run_date), slot)
    newest = data["snapshot_date"].max() if not data.empty else None
    if newest is None or newest <= last["data_end"]:
        return False, (f"no new labelled applications since {last['version']} "
                       f"(newest labelled application month {newest}); retraining would reproduce the same model")

    if C.is_scheduled_retrain(run_date):
        return True, f"scheduled retrain (every {C.RETRAIN_EVERY_MONTHS} months from go-live)"
    alerts = red_alerts_since(slot, last["trained_on"], run_date)
    if not alerts.empty:
        a = alerts.iloc[-1]
        return True, f"off-cycle retrain triggered by red alert: {a['metric']}={a['value']:.3f} on {a['monitoring_date']}"
    return False, f"not a scheduled month and no red alerts since {last['trained_on']}"


def main(slot, run_date):
    reg = R.load_registry()
    due, reason = retrain_decision(slot, run_date, reg)
    if not due:
        skip(f"[{slot}] no retrain on {run_date}: {reason}")
    print(f"[{slot}] retraining on {run_date}: {reason}")

    artefact = model_training.train_slot(slot, run_date)
    artefact["card"]["retrain_reason"] = reason
    artefact["card"]["data_end"] = artefact["oot_months"][-1]
    R.save_artefact(artefact)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slot", required=True, choices=list(C.MODEL_SLOTS))
    p.add_argument("--snapshotdate", required=True)
    a = p.parse_args()
    main(a.slot, a.snapshotdate)
