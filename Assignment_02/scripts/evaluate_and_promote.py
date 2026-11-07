"""
Promotion gate (champion / challenger) for every slot that produced a candidate this month.

    python scripts/evaluate_and_promote.py --snapshotdate 2024-03-01

The candidate and the current champion are both scored on the candidate's out-of-time months
(the most recent labelled applications, which neither model was trained on). The candidate goes
live only if its AUC is at least the champion's AUC minus PROMOTION_TOLERANCE; otherwise it is
recorded as rejected and the champion keeps serving. Every decision is written to the registry.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import enter_project_root, skip  # noqa: E402

enter_project_root()

from utils import config as C  # noqa: E402
from utils import ml_data  # noqa: E402
from utils import metrics as M  # noqa: E402
from utils import model_registry as R  # noqa: E402


def main(run_date):
    reg = R.load_registry()
    decided = 0
    for slot in C.MODEL_SLOTS:
        version = R.version_name(slot, run_date)
        if not os.path.exists(R.artefact_path(slot, version)):
            continue
        cand = R.load_artefact(slot, version)
        champ_entry = R.champion_at(reg, slot, run_date, strictly_before=True)

        data = ml_data.slot_population(ml_data.labelled_data(run_date), slot)
        oot = data[data["snapshot_date"].isin(cand["oot_months"])]
        y = oot["label"].astype(int)
        cand_auc = M.safe_auc(y, cand["model"].predict_proba(oot[cand["features"]].astype(float))[:, 1])

        if champ_entry is None:
            promote, champ_auc = True, None
            detail = f"first deployment; OOT AUC {cand_auc:.3f}"
        else:
            champ = R.load_artefact(slot, champ_entry["version"])
            champ_auc = M.safe_auc(y, champ["model"].predict_proba(oot[champ["features"]].astype(float))[:, 1])
            promote = cand_auc >= champ_auc - C.PROMOTION_TOLERANCE
            detail = (f"challenger {version} AUC {cand_auc:.3f} vs champion {champ_entry['version']} AUC {champ_auc:.3f} "
                      f"on {cand['oot_months'][0]}..{cand['oot_months'][-1]} applications (n={len(y)})")

        card = cand["card"]
        entry = {
            "version": version, "slot": slot, "algorithm": card["algorithm"], "trained_on": run_date,
            "data_end": card["data_end"], "artefact": R.artefact_path(slot, version), "card": R.card_path(slot, version),
            "retrain_reason": card.get("retrain_reason"), "oot_auc": card["metrics"]["oot_auc"],
            "decision_threshold": card["decision_threshold"],
            "promotion": {"challenger_auc": cand_auc, "champion_auc": champ_auc,
                          "champion_version": champ_entry["version"] if champ_entry else None, "detail": detail},
            "status": "champion" if promote else "rejected",
            "promoted_at": run_date if promote else None,
            "retired_at": None,
        }
        # re-running an old month must not undo what happened later (e.g. this version retired in a later month)
        previous = next((v for v in reg["slots"].get(slot, []) if v["version"] == version), None)
        if promote and previous and previous.get("retired_at") and previous["retired_at"] > run_date:
            entry["retired_at"], entry["status"] = previous["retired_at"], "retired"
        if champ_entry is not None:
            if promote:
                champ_entry["retired_at"], champ_entry["status"] = run_date, "retired"
            elif champ_entry.get("retired_at") == run_date:
                # re-running this month reversed an earlier promotion; a later retirement is left untouched
                champ_entry["retired_at"], champ_entry["status"] = None, "champion"
        R.upsert_version(reg, entry)
        R.log_event(reg, run_date, slot, "promoted" if promote else "rejected", version, detail)
        print(f"[{slot}] {'PROMOTED' if promote else 'REJECTED'}: {detail}")
        decided += 1

    if decided == 0:
        skip(f"no candidate models trained on {run_date}")
    R.save_registry(reg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    main(p.parse_args().snapshotdate)
