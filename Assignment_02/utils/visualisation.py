"""
Monitoring charts (PNG) and a single-page HTML report built from the gold monitoring tables
and the model registry. Re-generated after every monthly run, so it always shows the latest state.

Colour roles (validated categorical palette; status colours reserved for traffic lights):
  model_1 = blue, model_2 = orange, third series = aqua; good / warning / critical for thresholds.
"""
import base64
import html
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import config as C  # noqa: E402

PALETTE = {
    "model_1": "#2a78d6", "model_2": "#eb6834", "series_3": "#1baf7a",
    "good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "surface": "#fcfcfb", "shade": "#f0efec",
}
LABELS = {"model_1": "model_1 · full (with clickstream)", "model_2": "model_2 · core (no clickstream)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": PALETTE["axis"], "axes.labelcolor": PALETTE["ink2"],
    "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"], "axes.grid": True,
    "grid.color": PALETTE["grid"], "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": PALETTE["surface"], "axes.facecolor": PALETTE["surface"], "savefig.facecolor": PALETTE["surface"],
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def _dates(s):
    return pd.to_datetime(s)


def _month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))


def _titles(ax, title, subtitle):
    """Title + one-line subtitle, left-aligned to the plot area (kept clear of each other)."""
    ax.set_title("")
    ax.text(0, 1.11, title, transform=ax.transAxes, fontsize=13, fontweight="bold", color=PALETTE["ink"], ha="left", va="bottom")
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5, color=PALETTE["ink2"], ha="left", va="bottom")


def _legend_below(ax, ncol=3, handles=None):
    kw = {"handles": handles} if handles else {}
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.17), ncol=ncol, fontsize=8.5, **kw)


def _shade_outage(ax, outage_start):
    if outage_start is not None:
        ax.axvspan(_dates(outage_start) - pd.Timedelta(days=15), _dates(C.FEATURE_END) + pd.Timedelta(days=15),
                   color=PALETTE["shade"], zorder=0, lw=0)


def _end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=9, color=PALETTE["ink2"],
                fontweight="bold")
    ax.plot([x], [y], "o", color=color, ms=6, zorder=5)


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _pivot(mon, group, metric):
    d = mon[(mon["metric_group"] == group) & (mon["metric"] == metric)]
    return d.pivot_table(index="cohort_date", columns="model_slot", values="value", aggfunc="mean").sort_index()


def outage_start(mon):
    cov = mon[(mon["model_slot"] == C.PRODUCTION_NAME) & (mon["metric"] == "clickstream_coverage") & (mon["status"] == "red")]
    return cov["cohort_date"].min() if not cov.empty else None


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def chart_performance(mon, reg, out_dir, outage):
    auc = _pivot(mon, "performance", "auc")
    if auc.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _shade_outage(ax, outage)
    for slot in ["model_1", "model_2"]:
        if slot in auc:
            s = auc[slot].dropna()
            ax.plot(_dates(s.index), s.values, color=PALETTE[slot], marker="o", label=LABELS[slot])
            _end_label(ax, _dates(s.index[-1]), s.values[-1], f"{s.values[-1]:.2f}", PALETTE[slot])
    # alert line: production expected AUC minus the red threshold
    ref = mon[(mon["model_slot"] == C.PRODUCTION_NAME) & (mon["metric"] == "auc")].set_index("cohort_date")["reference_value"].sort_index()
    if not ref.empty:
        ax.step(_dates(ref.index), (ref - C.AUC_DROP_RED).values, where="mid", color=PALETTE["critical"], lw=1.2, ls="--",
                label=f"Alert line: expected AUC − {C.AUC_DROP_RED} (production)")
    for slot in ["model_1", "model_2"]:
        for v in reg["slots"].get(slot, []):
            if v.get("promoted_at") and v["promoted_at"] <= C.FEATURE_END:
                ax.plot(_dates(v["promoted_at"]), 0.712 if slot == "model_1" else 0.694, marker="^",
                        color=PALETTE[slot], ms=7, ls="none", label="_nolegend_")
    ax.plot([], [], marker="^", color=PALETTE["muted"], ls="none", ms=7, label="New version promoted (live from that month)")
    if outage:
        ax.text(_dates(outage), 0.985, "  Clickstream stops: all applicants routed to model_2", fontsize=9, color=PALETTE["ink2"], va="top")
    ax.set_ylim(0.68, 1.0)
    ax.set_ylabel("AUC (0.5 = coin flip, 1.0 = perfect)")
    ax.set_xlabel("Application month (labels observed 6 months later)")
    _month_axis(ax)
    _legend_below(ax, ncol=2)
    _titles(ax, "Model performance by application month", "AUC of each model on the applicants it scored, once their 6-month default label is known")
    return _save(fig, os.path.join(out_dir, "01_performance_auc.png"))


def chart_stability(mon, out_dir, outage):
    psi = _pivot(mon, "stability", "psi_score")
    if psi.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.3))
    _shade_outage(ax, outage)
    ax.axhspan(C.PSI_AMBER, C.PSI_RED, color=PALETTE["warning"], alpha=0.12, lw=0)
    ax.axhspan(C.PSI_RED, 1, color=PALETTE["critical"], alpha=0.08, lw=0)
    ax.text(_dates(psi.index[0]), C.PSI_AMBER + 0.005, " ⚠ watch (0.10–0.25)", fontsize=8.5, color=PALETTE["ink2"], va="bottom")
    ax.text(_dates(psi.index[0]), C.PSI_RED + 0.005, " ✖ act (> 0.25)", fontsize=8.5, color=PALETTE["ink2"], va="bottom")
    for slot in ["model_1", "model_2"]:
        if slot in psi:
            s = psi[slot].dropna()
            ax.plot(_dates(s.index), s.values, color=PALETTE[slot], marker="o", label=LABELS[slot])
            _end_label(ax, _dates(s.index[-1]), s.values[-1], f"{s.values[-1]:.3f}", PALETTE[slot])
    ax.set_ylim(0, max(0.3, float(np.nanmax(psi.values)) * 1.2))
    ax.set_ylabel("PSI of predicted default probability")
    ax.set_xlabel("Application month (available immediately)")
    _month_axis(ax)
    ax.legend(loc="upper right", fontsize=8.5)
    _titles(ax, "Score stability vs each model's training data", "Population stability index: how far this month's score distribution has moved from training (< 0.10 = stable)")
    return _save(fig, os.path.join(out_dir, "02_stability_psi.png"))


def chart_calibration(mon, out_dir, outage):
    prod = mon[(mon["model_slot"] == C.PRODUCTION_NAME) & (mon["metric_group"] == "performance")]
    if prod.empty:
        return None
    p = prod.pivot_table(index="cohort_date", columns="metric", values="value").sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.3))
    _shade_outage(ax, outage)
    x = _dates(p.index)
    ax.plot(x, p["actual_default_rate"] * 100, color=PALETTE["model_2"], marker="o", label="Actual default rate")
    ax.plot(x, p["mean_predicted_pd"] * 100, color=PALETTE["model_1"], marker="s", ls="--", label="Mean predicted default probability")
    _end_label(ax, x[-1], p["actual_default_rate"].iloc[-1] * 100, f"actual {p['actual_default_rate'].iloc[-1]:.1%}", PALETTE["model_2"], dy=-8)
    _end_label(ax, x[-1], p["mean_predicted_pd"].iloc[-1] * 100, f"predicted {p['mean_predicted_pd'].iloc[-1]:.1%}", PALETTE["model_1"], dy=8)
    ax.set_ylim(0, 45)
    ax.set_ylabel("% of applicants")
    ax.set_xlabel("Application month")
    _month_axis(ax)
    ax.legend(loc="lower left", fontsize=8.5)
    gap = (p["mean_predicted_pd"] - p["actual_default_rate"]).abs().max()
    _titles(ax, "Calibration: predicted vs actual default rate (production)", f"Largest monthly gap {gap:.1%} (alert at {C.CALIBRATION_GAP_AMBER:.0%}); a well-calibrated score keeps cut-offs meaningful")
    return _save(fig, os.path.join(out_dir, "03_calibration.png"))


def chart_routing(mon, out_dir):
    prod = mon[(mon["model_slot"] == C.PRODUCTION_NAME) & (mon["metric_group"] == "stability")]
    if prod.empty:
        return None
    p = prod.pivot_table(index="cohort_date", columns="metric", values="value").sort_index()
    n1 = (p["n_scored"] * p["share_routed_model_1"]).round()
    n2 = p["n_scored"] - n1
    fig, ax = plt.subplots(figsize=(10, 4.3))
    x = _dates(p.index)
    w = 22
    ax.bar(x, n1, width=w, color=PALETTE["model_1"], label=LABELS["model_1"], edgecolor=PALETTE["surface"], linewidth=1)
    ax.bar(x, n2, width=w, bottom=n1, color=PALETTE["model_2"], label=LABELS["model_2"], edgecolor=PALETTE["surface"], linewidth=1)
    for xi, a, b in zip(x, n1, n2):
        ax.text(xi, a + b + 8, f"{int(a + b)}", ha="center", fontsize=8, color=PALETTE["ink2"])
    ax.set_ylim(0, p["n_scored"].max() * 1.18)
    ax.set_ylabel("Applications decided")
    ax.set_xlabel("Application month")
    _month_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    _titles(ax, "Which model made the production decision", "Routing on data availability: applicants with clickstream history go to model_1, everyone else to model_2")
    return _save(fig, os.path.join(out_dir, "04_routing_volume.png"))


def chart_business(mon, out_dir, outage):
    prod = mon[(mon["model_slot"] == C.PRODUCTION_NAME) & (mon["metric_group"] == "performance")]
    if prod.empty:
        return None
    p = prod.pivot_table(index="cohort_date", columns="metric", values="value").sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.3))
    _shade_outage(ax, outage)
    x = _dates(p.index)
    series = [("approval_rate", "Approval rate", PALETTE["model_1"], "o", "-"),
              ("actual_default_rate", "Default rate if every applicant were approved", PALETTE["muted"], None, ":"),
              ("bad_rate_approved", "Default rate among approved applicants", PALETTE["model_2"], "o", "-")]
    for key, label, color, marker, ls in series:
        ax.plot(x, p[key] * 100, color=color, marker=marker, ls=ls, label=label)
        _end_label(ax, x[-1], p[key].iloc[-1] * 100, f"{p[key].iloc[-1]:.0%}", color)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of applicants")
    ax.set_xlabel("Application month")
    _month_axis(ax)
    _legend_below(ax, ncol=3)
    _titles(ax, "Business outcome of the cut-off (production)", f"Approved applicants default at {p['bad_rate_approved'].mean():.0%} on average vs "
                   f"{p['actual_default_rate'].mean():.0%} for all applicants")
    return _save(fig, os.path.join(out_dir, "05_business_outcomes.png"))


def chart_feature_drift(mon, out_dir):
    d = mon[(mon["metric_group"] == "stability") & (mon["metric"].str.startswith("csi__"))].copy()
    if d.empty:
        return None
    d["feature"] = d["metric"].str.replace("csi__", "", regex=False)
    slots = [s for s in ["model_1", "model_2"] if s in d["model_slot"].unique()]
    fig, axes = plt.subplots(1, len(slots), figsize=(12, 5.2), gridspec_kw={"width_ratios": [1] * len(slots)})
    axes = np.atleast_1d(axes)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.5)
    for ax, slot in zip(axes, slots):
        s = d[d["model_slot"] == slot]
        latest = s[s["cohort_date"] == s["cohort_date"].max()]
        order = latest.sort_values("value", ascending=False)["feature"].tolist()
        order += [f for f in s["feature"].unique() if f not in order]
        order = order[:C.TOP_FEATURES_MONITORED]
        mat = s.pivot_table(index="feature", columns="cohort_date", values="value").reindex(order)
        ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if v > 0.3 else PALETTE["ink"])
        ax.set_yticks(range(len(order)), order, fontsize=8)
        ax.set_xticks(range(mat.shape[1]), [pd.to_datetime(c).strftime("%b %y") for c in mat.columns], fontsize=7, rotation=90)
        ax.grid(False)
        ax.set_title(LABELS[slot], fontsize=10)
    fig.suptitle("Feature drift (CSI) of each model's top features vs training", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.905, "Same index as PSI, per feature. Darker = larger shift; > 0.25 is flagged red", fontsize=9.5, color=PALETTE["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, os.path.join(out_dir, "06_feature_drift_csi.png"))


def chart_model_timeline(reg, out_dir):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_axisbelow(True)
    end = _dates(C.FEATURE_END) + pd.Timedelta(days=31)
    rows = {"model_1": 1, "model_2": 0}
    for slot, y in rows.items():
        for v in reg["slots"].get(slot, []):
            if v.get("promoted_at"):
                start = _dates(v["promoted_at"])
                stop = _dates(v["retired_at"]) if v.get("retired_at") else end
                ax.barh(y, (stop - start).days, left=start, height=0.5, color=PALETTE[slot],
                        edgecolor=PALETTE["surface"], linewidth=2, zorder=3)
                ax.text(start + (stop - start) / 2, y, f"{v['algorithm'].replace('logistic_regression', 'LogReg').replace('xgboost', 'XGB')}\n{v['oot_auc']:.2f}",
                        ha="center", va="center", fontsize=7.5, color="#ffffff", zorder=4)
            else:
                ax.plot(_dates(v["trained_on"]), y + 0.38, marker="x", color=PALETTE["critical"], ms=7, mew=2, zorder=5)
    ax.plot([], [], marker="x", color=PALETTE["critical"], ls="none", mew=2, label="Challenger rejected by promotion gate")
    ax.set_yticks(list(rows.values()), [LABELS[s] for s in rows], fontsize=9)
    ax.set_ylim(-0.6, 1.7)
    _month_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=8.5)
    _titles(ax, "Live model versions (bar label: algorithm and out-of-time AUC)", f"Retrained every {C.RETRAIN_EVERY_MONTHS} months from go-live; a challenger replaces the champion only if it is at least as good on the latest labelled months")
    return _save(fig, os.path.join(out_dir, "07_model_timeline.png"))


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def _img(path):
    with open(path, "rb") as f:
        return f'<img alt="{os.path.basename(path)}" src="data:image/png;base64,{base64.b64encode(f.read()).decode()}">'


def _table(df, fmt=None):
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt and isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                v = fmt[c].format(v)
            cls = f' class="st-{v}"' if c == "status" else ""
            cells.append(f"<td{cls}>{html.escape(str(v)) if not pd.isna(v) else ''}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_report(mon, reg, charts, out_path, run_date):
    perf = mon[(mon["model_slot"] == C.PRODUCTION_NAME) & (mon["metric_group"] == "performance")]
    perf_t = perf.pivot_table(index="cohort_date", columns="metric", values="value").reset_index()
    cols = [c for c in ["cohort_date", "n_labelled", "auc", "gini", "ks", "actual_default_rate", "mean_predicted_pd",
                        "approval_rate", "bad_rate_approved", "bad_rate_declined"] if c in perf_t.columns]
    perf_t = perf_t[cols]
    alerts = mon[mon["status"].isin(["red", "amber"])][["monitoring_date", "model_slot", "model_version", "cohort_date",
                                                         "metric", "value", "reference_value", "status"]]
    alerts = alerts.sort_values(["monitoring_date", "model_slot", "metric"])
    events = pd.DataFrame(reg.get("events", []))
    pct = "{:.1%}"
    sections = [f"<h2>{html.escape(t)}</h2>{_img(p)}" for t, p in charts if p]
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Loan default model - monitoring report</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#f9f9f7;color:#0b0b0b;margin:0;padding:24px 32px;max-width:1180px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} p{{color:#52514e;margin:4px 0}}
img{{width:100%;max-width:1100px;border:1px solid rgba(11,11,11,.1);border-radius:6px;background:#fcfcfb}}
table{{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;background:#fcfcfb;margin:6px 0}}
th,td{{border-bottom:1px solid #e1e0d9;padding:4px 8px;text-align:right}} th{{color:#52514e;font-weight:600}}
td:first-child,th:first-child{{text-align:left}}
.st-red{{color:#d03b3b;font-weight:600}} .st-red::before{{content:"✖ "}}
.st-amber{{color:#9a6a00;font-weight:600}} .st-amber::before{{content:"⚠ "}}
details{{margin:8px 0}}
</style></head><body>
<h1>Loan default model - monitoring report</h1>
<p>Generated by the Airflow DAG after the {html.escape(run_date)} run. Go-live {C.GO_LIVE_DATE}; retrain every {C.RETRAIN_EVERY_MONTHS} months;
label = {C.LABEL_DEF} (observed 6 months after application). Thresholds: PSI {C.PSI_AMBER}/{C.PSI_RED}, AUC drop {C.AUC_DROP_AMBER}/{C.AUC_DROP_RED},
calibration gap {C.CALIBRATION_GAP_AMBER}/{C.CALIBRATION_GAP_RED}.</p>
{''.join(sections)}
<h2>Production performance by application month (table view)</h2>
{_table(perf_t, {c: pct if 'rate' in c else '{:.3f}' for c in cols if c not in ('cohort_date', 'n_labelled')} | {'n_labelled': '{:.0f}'})}
<h2>Model registry events</h2>
{_table(events) if not events.empty else '<p>No events yet.</p>'}
<details><summary><b>All amber / red alerts ({len(alerts)})</b></summary>
{_table(alerts, {'value': '{:.3f}', 'reference_value': '{:.3f}'})}</details>
</body></html>"""
    with open(out_path, "w") as f:
        f.write(doc)
    return out_path
