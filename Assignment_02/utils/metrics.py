"""Model performance and stability metrics used in training and monitoring."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def ks_statistic(y, p):
    """Kolmogorov-Smirnov: largest gap between the score distributions of defaulters and non-defaulters."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ks_threshold(y, p):
    """Cut-off where the KS gap is largest (separates good and bad applicants best)."""
    fpr, tpr, thr = roc_curve(np.asarray(y), p)
    i = int(np.argmax(tpr - fpr))
    return float(min(max(thr[i], 0.0), 1.0))


def classification_summary(y, p, threshold):
    """Business view of a cut-off: decline if predicted default probability >= threshold."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    declined = p >= threshold
    approved = ~declined
    tp = int(((y == 1) & declined).sum())
    fp = int(((y == 0) & declined).sum())
    fn = int(((y == 1) & approved).sum())
    tn = int(((y == 0) & approved).sum())
    n = len(y)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "accuracy": (tp + tn) / n if n else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else float("nan"),
        "approval_rate": approved.mean() if n else float("nan"),
        "bad_rate_approved": y[approved].mean() if approved.any() else float("nan"),
        "bad_rate_declined": y[declined].mean() if declined.any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# stability: population stability index (PSI) / characteristic stability index (CSI)
# ---------------------------------------------------------------------------
EPS = 1e-4


def make_bins(reference_values, n_bins=10):
    """Decile edges of the reference (training) distribution; missing values get their own bin."""
    v = np.asarray(reference_values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return [-np.inf, np.inf]
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if len(edges) <= 2:   # (near) constant or binary feature
        edges = np.unique(v)
        edges = np.concatenate([[-np.inf], (edges[:-1] + edges[1:]) / 2, [np.inf]]) if len(edges) > 1 else np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    return [float(e) for e in edges]


def bin_proportions(values, edges):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return []
    missing = np.isnan(v)
    counts = np.histogram(v[~missing], bins=np.asarray(edges, dtype=float))[0]
    return [float(c / n) for c in counts] + [float(missing.sum() / n)]


def psi(reference_props, actual_props):
    """Sum over bins of (actual - expected) * ln(actual / expected). <0.1 stable, 0.1-0.25 watch, >0.25 shift."""
    r = np.clip(np.asarray(reference_props, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual_props, dtype=float), EPS, None)
    return float(np.sum((a - r) * np.log(a / r)))


def status_high_is_bad(value, amber, red):
    if value is None or np.isnan(value):
        return "info"
    return "red" if value > red else "amber" if value > amber else "green"
