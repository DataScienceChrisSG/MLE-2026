"""
Model bank + registry.

model_bank/<slot>/<version>.pkl   the artefact (preprocessing + model in one sklearn pipeline, plus metadata)
model_bank/<slot>/<version>.json  the model card (human-readable metadata, no model object)
model_bank/model_registry.json    every version ever trained, its status and when it was live

Status: champion (serving), retired (was champion, replaced), rejected (failed the promotion gate).
A version is live from promoted_at (inclusive) until retired_at (exclusive), so any month can be
re-scored with the model that was actually live at that time (point-in-time lookup, rollback).
"""
import os
import pickle

from utils import config as C
from utils.common import read_json, write_json


def version_name(slot, trained_on):
    return f"{slot}_{C.partition_suffix(trained_on)}"


def artefact_path(slot, version):
    return os.path.join(C.MODEL_BANK_DIR, slot, f"{version}.pkl")


def card_path(slot, version):
    return os.path.join(C.MODEL_BANK_DIR, slot, f"{version}.json")


def save_artefact(artefact):
    slot, version = artefact["slot"], artefact["version"]
    os.makedirs(os.path.join(C.MODEL_BANK_DIR, slot), exist_ok=True)
    with open(artefact_path(slot, version), "wb") as f:
        pickle.dump(artefact, f)
    write_json(artefact["card"], card_path(slot, version))
    print("saved model artefact:", artefact_path(slot, version))


def load_artefact(slot, version):
    with open(artefact_path(slot, version), "rb") as f:
        return pickle.load(f)


def load_registry():
    reg = read_json(C.REGISTRY_PATH)
    if reg is None:
        reg = {"slots": {s: [] for s in C.MODEL_SLOTS}, "events": []}
    return reg


def save_registry(reg):
    reg["policy"] = {
        "go_live": C.GO_LIVE_DATE,
        "retrain_every_months": C.RETRAIN_EVERY_MONTHS,
        "promotion_rule": f"challenger AUC on the latest {C.OOT_MONTHS} labelled months >= champion AUC - {C.PROMOTION_TOLERANCE}",
        "routing": "model_1 if the applicant has clickstream history, else model_2",
    }
    for s in reg["slots"]:
        reg["slots"][s].sort(key=lambda v: v["trained_on"])
    reg["events"].sort(key=lambda e: (e["date"], e["slot"]))
    write_json(reg, C.REGISTRY_PATH)


def champion_at(reg, slot, date_str, strictly_before=False):
    """The version live on date_str (or live just before it, when deciding a promotion on date_str)."""
    for v in reg["slots"].get(slot, []):
        if v.get("promoted_at") is None:
            continue
        starts_ok = v["promoted_at"] < date_str if strictly_before else v["promoted_at"] <= date_str
        retired = v.get("retired_at")
        ends_ok = retired is None or (retired >= date_str if strictly_before else retired > date_str)
        if starts_ok and ends_ok:
            return v
    return None


def upsert_version(reg, entry):
    reg["slots"].setdefault(entry["slot"], [])
    reg["slots"][entry["slot"]] = [v for v in reg["slots"][entry["slot"]] if v["version"] != entry["version"]]
    reg["slots"][entry["slot"]].append(entry)


def log_event(reg, date_str, slot, event, version, detail):
    # re-running a month replaces that month's events (idempotent)
    reg["events"] = [e for e in reg["events"] if not (e["date"] == date_str and e["slot"] == slot and e["event"] == event)]
    reg["events"].append({"date": date_str, "slot": slot, "event": event, "version": version, "detail": detail})
