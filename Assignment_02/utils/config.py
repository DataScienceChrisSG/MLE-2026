"""
Central configuration for the loan default ML pipeline.

Every script reads its paths, dates and thresholds from here, so a policy change
(e.g. the retraining cadence or a monitoring threshold) is a one-line edit.
All paths are relative to the project root, which every script uses as its working directory.
"""
from datetime import datetime
from dateutil.relativedelta import relativedelta


# ---------------------------------------------------------------------------
# raw sources (treated as the source systems)
# ---------------------------------------------------------------------------
SOURCES = {
    "loan_daily": "data/lms_loan_daily.csv",
    "attributes": "data/features_attributes.csv",
    "financials": "data/features_financials.csv",
    "clickstream": "data/feature_clickstream.csv",
}

# ---------------------------------------------------------------------------
# datamart layout (medallion)
# ---------------------------------------------------------------------------
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
        "model_predictions": "datamart/gold/model_predictions/",
        "model_monitoring": "datamart/gold/model_monitoring/",
    },
}
MODEL_BANK_DIR = "model_bank/"
REGISTRY_PATH = "model_bank/model_registry.json"
REPORTS_DIR = "reports/"

# ---------------------------------------------------------------------------
# data availability (from EDA in Assignment 1)
# ---------------------------------------------------------------------------
# Loan applications (= feature snapshots) run Jan 2023 - Jan 2025.
FEATURE_START, FEATURE_END = "2023-01-01", "2025-01-01"
# Clickstream is only recorded up to Dec 2024 (and only for applicants up to Jun 2024).
CLICKSTREAM_START, CLICKSTREAM_END = "2023-01-01", "2024-12-01"
# Labels (30 dpd at month-on-book 6) are observed Jul 2023 - Jul 2025.
LABEL_DPD, LABEL_MOB = 30, 6
LABEL_DEF = f"{LABEL_DPD}dpd_{LABEL_MOB}mob"
# The monthly pipeline runs until the last label matures. LMS months after that are not needed.
PIPELINE_START, PIPELINE_END = "2023-01-01", "2025-07-01"
LMS_START, LMS_END = PIPELINE_START, PIPELINE_END

# date range in which each table has data; outside it the task is skipped
TABLE_DATE_RANGES = {
    "loan_daily": (LMS_START, LMS_END),
    "attributes": (FEATURE_START, FEATURE_END),
    "financials": (FEATURE_START, FEATURE_END),
    # processed for every application month so the gold job can always read "all months <= application"
    "clickstream": (FEATURE_START, FEATURE_END),
    "label_store": (PIPELINE_START, PIPELINE_END),
    "feature_store": (FEATURE_START, FEATURE_END),
}

# ---------------------------------------------------------------------------
# model lifecycle policy (agreed design)
# ---------------------------------------------------------------------------
GO_LIVE_DATE = "2024-01-01"          # first model trained and deployed; scoring starts this month
RETRAIN_EVERY_MONTHS = 2             # scheduled retrain cadence, counted from go-live
OOT_MONTHS = 2                       # most recent labelled application months held out as out-of-time test
TEST_SIZE = 0.2                      # random stratified test split of the remaining (older) months
RANDOM_SEED = 88
SEARCH_ITERATIONS = 20               # randomised hyper-parameter search per algorithm
CV_FOLDS = 3
ALGORITHM_TIE_TOLERANCE = 0.005      # if AUCs are this close, prefer the simpler model (logistic regression)
PROMOTION_TOLERANCE = 0.0            # challenger must be at least as good as the champion on the latest labelled months

# Two model slots with routing on clickstream availability
MODEL_SLOTS = {
    "model_1": {
        "label": "Full model (with clickstream)",
        "feature_set": "full",
        "population": "applicants with clickstream history",
    },
    "model_2": {
        "label": "Core model (no clickstream)",
        "feature_set": "core",
        "population": "all applicants (production score when clickstream is missing)",
    },
}
PRODUCTION_NAME = "production"       # routed decision: model_1 if has_clickstream else model_2

# ---------------------------------------------------------------------------
# monitoring thresholds (traffic lights)
# ---------------------------------------------------------------------------
PSI_AMBER, PSI_RED = 0.10, 0.25                    # population stability index
AUC_DROP_AMBER, AUC_DROP_RED = 0.03, 0.05          # drop vs the model's out-of-time AUC at training
CALIBRATION_GAP_AMBER, CALIBRATION_GAP_RED = 0.05, 0.10   # |mean predicted PD - actual default rate|
MISSING_RATE_INCREASE_RED = 0.20                    # feature missing rate up by 20+ percentage points vs training
TOP_FEATURES_MONITORED = 10                         # CSI tracked for the model's 10 most important features


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def to_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")


def add_months(date_str, n):
    return (to_date(date_str) + relativedelta(months=n)).strftime("%Y-%m-%d")


def months_between(start_str, end_str):
    a, b = to_date(start_str), to_date(end_str)
    return (b.year - a.year) * 12 + (b.month - a.month)


def in_range(date_str, start_str, end_str):
    return start_str <= date_str <= end_str


def partition_suffix(date_str):
    return date_str.replace("-", "_")


def predictions_path(slot, date_str):
    return f"{DIRS['gold']['model_predictions']}{slot}/{slot}_predictions_{partition_suffix(date_str)}.parquet"


def monitoring_path(slot, date_str):
    return f"{DIRS['gold']['model_monitoring']}{slot}/{slot}_monitoring_{partition_suffix(date_str)}.parquet"


def is_scheduled_retrain(snapshot_date_str):
    """Go-live month and every RETRAIN_EVERY_MONTHS after it (while new labels keep arriving)."""
    if snapshot_date_str < GO_LIVE_DATE or snapshot_date_str > PIPELINE_END:
        return False
    return months_between(GO_LIVE_DATE, snapshot_date_str) % RETRAIN_EVERY_MONTHS == 0
