"""
Loan default ML pipeline - one monthly DAG that runs end to end:

  data pipeline (bronze -> silver -> gold feature & label stores)
  -> retrain if due (model_1 full, model_2 core) -> champion/challenger promotion
  -> inference (model_1, model_2, routed production decision)
  -> monitoring (stability now, performance once labels mature 6 months later)
  -> monitoring charts + HTML report

Every task calls one script with --snapshotdate {{ ds }}. A script exits with code 99 when there is
nothing to do for that month (e.g. no applications after Jan 2025); Airflow then shows it as SKIPPED.
"""
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["loan_daily", "attributes", "financials", "clickstream"]
SLOTS = ["model_1", "model_2"]

DOC = """
### Loan default ML pipeline (CS611 Assignment 2)

**How to run:** the DAG is un-paused on start-up and back-fills every month from **Jan 2023 to Jul 2025**
(31 runs, one at a time, in date order). Watch progress in *Grid* view. To re-run everything, clear the runs
(Browse > DAG Runs) or trigger a back-fill from the CLI:
`airflow dags backfill loan_default_ml_pipeline -s 2023-01-01 -e 2025-07-01 --reset-dagruns -y`.
A manual trigger outside that window completes with every task *skipped* (no source data).

**Timeline:** data only (Jan-Dec 2023) -> go-live and first models (Jan 2024) -> retrain every 2 months ->
clickstream stops for new applicants (Jul 2024, routing switches to model_2) -> last applications (Jan 2025)
-> last labels mature (Jul 2025).

**Outputs** (project folder): `datamart/gold/model_predictions/`, `datamart/gold/model_monitoring/`,
`model_bank/` (artefacts, model cards, `model_registry.json`), `reports/monitoring_report.html`.
"""


def task(task_id, script, args="", trigger_rule=TriggerRule.ALL_SUCCESS):
    return BashOperator(
        task_id=task_id,
        bash_command=f'cd "{PROJECT_ROOT}" && python scripts/{script} {args} --snapshotdate "{{{{ ds }}}}"',
        trigger_rule=trigger_rule,
    )


default_args = {
    "owner": "ml-engineering",
    "depends_on_past": True,           # a month only runs if the previous month succeeded (or was skipped)
    "retries": 2,
    "retry_delay": timedelta(seconds=30),
}

with DAG(
    dag_id="loan_default_ml_pipeline",
    description="Monthly loan default pipeline: datamart -> train/promote -> inference -> monitoring",
    doc_md=DOC,
    default_args=default_args,
    schedule="0 0 1 * *",              # 00:00 on the 1st of every month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 7, 1),
    catchup=True,
    max_active_runs=1,                 # months build on each other (clickstream history, model registry)
    max_active_tasks=1,                # one task at a time: Spark jobs never write to the shared folder concurrently
    is_paused_upon_creation=False,
    tags=["cs611", "loan-default", "medallion", "mlops"],
) as dag:

    start = EmptyOperator(task_id="start")
    check_source_data = task("check_source_data", "check_source_data.py")
    start >> check_source_data

    # ---------------- data pipeline (Assignment 1 code) ----------------
    silver = {}
    for src in SOURCES:
        b = task(f"bronze_{src}", "data_pipeline.py", f"--layer bronze --table {src}")
        s = task(f"silver_{src}", "data_pipeline.py", f"--layer silver --table {src}")
        check_source_data >> b >> s
        silver[src] = s

    gold_label_store = task("gold_label_store", "data_pipeline.py", "--layer gold --table label_store")
    gold_feature_store = task("gold_feature_store", "data_pipeline.py", "--layer gold --table feature_store")
    silver["loan_daily"] >> gold_label_store
    [silver[s] for s in SOURCES] >> gold_feature_store

    data_pipeline_completed = EmptyOperator(task_id="data_pipeline_completed", trigger_rule=TriggerRule.NONE_FAILED)
    [gold_label_store, gold_feature_store] >> data_pipeline_completed

    # ---------------- model training & promotion ----------------
    train = {slot: task(f"train_{slot}", "train_model.py", f"--slot {slot}") for slot in SLOTS}
    evaluate_and_promote = task("evaluate_and_promote", "evaluate_and_promote.py",
                                trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    model_training_completed = EmptyOperator(task_id="model_training_completed", trigger_rule=TriggerRule.NONE_FAILED)
    data_pipeline_completed >> list(train.values()) >> evaluate_and_promote >> model_training_completed

    # ---------------- inference ----------------
    infer = {slot: task(f"{slot}_inference", "model_inference.py", f"--slot {slot}") for slot in SLOTS}
    production_routing = task("production_routing", "model_inference.py", "--slot production",
                              trigger_rule=TriggerRule.NONE_FAILED)
    model_inference_completed = EmptyOperator(task_id="model_inference_completed", trigger_rule=TriggerRule.NONE_FAILED)
    model_training_completed >> list(infer.values()) >> production_routing >> model_inference_completed

    # ---------------- monitoring ----------------
    monitors = [task(f"{s}_monitoring", "model_monitoring.py", f"--slot {s}") for s in SLOTS + ["production"]]
    model_monitoring_completed = EmptyOperator(task_id="model_monitoring_completed", trigger_rule=TriggerRule.NONE_FAILED)
    model_inference_completed >> monitors >> model_monitoring_completed

    visualise_monitoring = task("visualise_monitoring", "visualise_monitoring.py")
    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED)
    model_monitoring_completed >> visualise_monitoring >> end
