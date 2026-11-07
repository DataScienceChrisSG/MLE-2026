"""
Reading the gold feature and label stores for the ML tasks (pandas - the data is small).

Feature store: one row per loan application, snapshot_date = application date.
Label store:   one row per loan, snapshot_date = date the label is observed (application + 6 months).
They join 1:1 on loan_id.
"""
import os
import glob

import pandas as pd

from utils import config as C

KEYS = ["loan_id", "Customer_ID", "snapshot_date"]
CLICKSTREAM_FEATURES = [f"fe_{i}_avg" for i in range(1, 21)] + ["clickstream_months_observed", "has_clickstream"]


def _partition_date(path, prefix):
    # gold_feature_store_2024_01_01.parquet -> 2024-01-01
    return os.path.basename(path)[len(prefix):-len(".parquet")].replace("_", "-")


def _read_gold(folder, prefix, first=None, last=None):
    frames = []
    for path in sorted(glob.glob(os.path.join(folder, f"{prefix}*.parquet"))):
        d = _partition_date(path, prefix)
        if (first is None or d >= first) and (last is None or d <= last):
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.strftime("%Y-%m-%d")
    return df


def load_features(first=None, last=None):
    return _read_gold(C.DIRS["gold"]["feature_store"], "gold_feature_store_", first, last)


def load_labels(first=None, last=None):
    return _read_gold(C.DIRS["gold"]["label_store"], "gold_label_store_", first, last)


def feature_columns(df, feature_set):
    cols = [c for c in df.columns if c not in KEYS + ["label", "label_def", "label_date"]]
    if feature_set == "core":
        cols = [c for c in cols if c not in CLICKSTREAM_FEATURES]
    return cols


def slot_population(df, slot):
    """model_1 only ever sees applicants with clickstream history; model_2 sees everyone."""
    if C.MODEL_SLOTS[slot]["feature_set"] == "full":
        return df[df["has_clickstream"] == 1]
    return df


def labelled_data(as_of):
    """
    Every application whose label has been observed by `as_of` (label date <= as_of).
    Because labels mature 6 months after application, this is applications up to as_of - 6 months:
    no information from after the run date is used (no temporal leakage).
    """
    labels = load_labels(last=as_of)
    if labels.empty:
        return pd.DataFrame()
    labels = labels[labels["snapshot_date"] <= as_of].rename(columns={"snapshot_date": "label_date"})
    feats = load_features(last=as_of)
    df = feats.merge(labels[["loan_id", "label", "label_date"]], on="loan_id", how="inner")
    return df.sort_values(["snapshot_date", "loan_id"]).reset_index(drop=True)
