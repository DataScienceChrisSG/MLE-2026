# CS611 Assignment 2 – Loan default ML pipeline (Airflow)

An end-to-end, monthly ML pipeline for predicting, **at loan application**, whether an applicant will be
30+ days past due 6 months into the loan. It builds on the Assignment 1 medallion datamart and adds
training, a model registry, champion/challenger promotion, routed inference, monitoring and a monitoring report,
all orchestrated by one Airflow DAG that back-fills Jan 2023 – Jul 2025.

## Run it

```bash
docker-compose build
docker-compose up          # Airflow UI: http://localhost:8080  (admin / admin)
```

The DAG `loan_default_ml_pipeline` is un-paused on start-up and back-fills 31 monthly runs in order.
Outputs are written into this folder (the shipped `datamart/`, `model_bank/` and `reports/` already hold a complete run).
Linux hosts: `echo "AIRFLOW_UID=$(id -u)" > .env` before `docker-compose up`.

## Folder structure

```
dags/loan_default_ml_pipeline.py   the DAG (28 tasks per monthly run)
scripts/                           one entry point per task, each takes --snapshotdate YYYY-MM-DD
  check_source_data.py             source files present with expected columns
  data_pipeline.py                 bronze / silver / gold for one table and month (Assignment 1 code)
  train_model.py                   retrain a model slot if the policy says so
  evaluate_and_promote.py          champion / challenger promotion gate, registry update
  model_inference.py               score this month's applications (model_1, model_2, routed production)
  model_monitoring.py              stability (this month) and performance (labels matured this month)
  visualise_monitoring.py          charts + HTML report
utils/                             config, Assignment 1 processing code, training, registry, metrics, charts
data/                              raw source files
datamart/bronze|silver|gold        medallion tables; gold also holds model_predictions and model_monitoring
model_bank/                        <slot>/<version>.pkl (pipeline + metadata), <version>.json (model card), model_registry.json
reports/                           monitoring_report.html and monitoring/*.png
```

## What one monthly run does

1. **Data pipeline** – bronze (raw copy) → silver (typed, cleaned with fixed business rules) → gold
   feature store (one row per application, application-date information only) and label store (30 dpd at month 6).
2. **Retrain if due** – at go-live (Jan 2024), then every 2 months, or off-cycle after a red monitoring alert,
   and only if new labelled data exists. Two model slots:
   * `model_1` full (78 features incl. clickstream) – trained and used for applicants with clickstream history
   * `model_2` core (56 features, no clickstream) – trained on all applicants; serves everyone without clickstream
   Each retrain compares logistic regression and XGBoost (randomised search, 3-fold CV on AUC), picks the
   algorithm on a held-out test split (ties go to logistic regression) and keeps the latest 2 labelled months as out-of-time (OOT) data.
3. **Promotion gate** – the challenger replaces the champion only if its AUC on the OOT months is at least the champion's.
   Every decision is written to `model_bank/model_registry.json`; older versions are kept for rollback.
4. **Inference** – only this month's applications are scored, with the version live on that date.
   Production decision = `model_1` if the applicant has clickstream history, else `model_2`.
5. **Monitoring** – stability now (score PSI, feature CSI, missing rates, clickstream coverage, approval rate) and
   performance for the applications scored 6 months ago, whose labels have just matured (AUC/Gini/KS, calibration,
   precision/recall, default rate among approved). Traffic-light thresholds in `utils/config.py`.
6. **Report** – `reports/monitoring_report.html` and charts, regenerated every month.

## Key design decisions

| Decision | Why |
|---|---|
| Keep the Assignment 1 feature store; classmates' datamarts not adopted | Their pipelines fit imputation / capping on the full dataset (train-test contamination) or drop customers without clickstream. Their extra ratios were tested and added no out-of-time lift (±0.001 AUC). |
| Two models with routing on clickstream availability | Clickstream lifts AUC from ~0.80 to ~0.91, but stops for applicants from Jul 2024. A single clickstream model then over-predicts default (46% vs 29% actual) and approvals at a fixed cut-off fall from 67% to 38%. Routing avoids this without anyone having to react. |
| Go-live Jan 2024, score only new applications | Earlier months are needed to train (labels arrive 6 months late); scoring training months would flatter monitoring. |
| Retrain every 2 months | Back-tested monthly / 2-monthly / quarterly / 6-monthly / never: 0.863 / 0.862 / 0.861 / 0.858 / 0.853 mean monthly AUC. 2-monthly captures the early gains when data is scarce. |
| Monitoring split by label lag | Stability is known immediately; performance only when the 6-month label matures. |
| One task at a time; silver/gold files written to a temporary folder, verified, then swapped in | The project folder is shared with the host. Writing in place and running several Spark jobs at once let a just-written file go missing for the next task on Docker Desktop (macOS). A task now only reports success once its file is verified, and failed tasks retry twice. |
| Preprocessing inside one sklearn pipeline | Fitted on training data only, travels with the model: no contamination, no train/serve skew. |

## Model governance and refresh SOP

* **Scheduled refresh:** every 2 months on all newly labelled applications; challenger must beat or match the champion on the latest labelled months.
* **Off-cycle refresh triggers:** red AUC drop (> 0.05 below the model's OOT AUC), red calibration gap (> 10 pts) or red score PSI (> 0.25).
* **Data incidents:** a red clickstream-coverage alert means an upstream outage; routing falls back to `model_2` automatically, the data owner is notified, no retrain needed.
* **Feature drift (CSI) red but score stable and AUC fine:** investigate, document, no action (e.g. clickstream averages naturally tighten as customers build longer histories).
* **Every version** has a model card (data window, features, hyper-parameters, metrics, threshold, library versions) and stays in the bank for rollback; the registry records when each version was live.
* **In production** a model-risk reviewer signs off each promotion; here the gate is automated and logged.

## Deployment options

* **Batch (implemented):** monthly Airflow scoring of applications into a gold table – simple, auditable, suited to back-testing.
* **Real-time API:** the same artefact behind a REST endpoint scoring each application as it arrives (the business need is a decision at application time).
* **Shadow / champion-challenger:** `model_2` already scores every applicant in parallel, so a new model can be compared on live traffic before it takes decisions.
