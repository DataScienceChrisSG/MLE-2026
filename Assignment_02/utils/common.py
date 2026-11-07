"""Shared helpers for the task entry points in scripts/."""
import os
import sys
import json
import glob
import shutil
import time
import uuid

import pandas as pd

# Airflow's BashOperator marks a task as SKIPPED (not failed) when the command exits with 99.
SKIP_EXIT_CODE = 99

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def enter_project_root():
    """All paths in utils/config.py are relative to the project root."""
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def skip(reason):
    print(f"SKIPPED: {reason}")
    sys.exit(SKIP_EXIT_CODE)


def get_spark(app_name):
    import pyspark
    spark = (
        pyspark.sql.SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.ui.enabled", "false")            # several tasks may run at once
        .config("spark.sql.shuffle.partitions", "8")    # small monthly partitions
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_parquet_folder(folder, pattern="*.parquet", columns=None):
    """Read every partition in a gold folder with pandas (ML tasks don't need Spark)."""
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True)


def write_parquet(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, index=False)
    print("saved to:", filepath, "| rows:", len(df))


def write_json(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, filepath)     # atomic: a crash never leaves a half-written file


def read_json(filepath, default=None):
    if not os.path.exists(filepath):
        return default
    with open(filepath) as f:
        return json.load(f)


def _is_complete_parquet_dir(path):
    """A Spark parquet output is complete when its folder has the _SUCCESS marker and at least one data file."""
    if not os.path.isdir(path):
        return False
    names = os.listdir(path)
    return "_SUCCESS" in names and any(n.endswith(".parquet") for n in names)


def safe_write_spark_parquet(df, filepath, attempts=3):
    """Write a Spark DataFrame so readers never see a missing or half-written folder.

    Spark's own overwrite deletes the target first and writes in place. On a folder shared with the host
    (Docker Desktop on macOS), that delete-then-recreate can leave the target missing for the next task.
    Instead: write to a temporary sibling folder, check it is complete, swap it into place, then check the
    final folder; raise (so Airflow retries the task) rather than report success without the file.
    """
    parent, name = os.path.split(filepath.rstrip("/"))
    os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(parent, f"_tmp_{name}_{uuid.uuid4().hex[:8]}")    # not matched by the *.parquet readers
    df.write.mode("overwrite").parquet(tmp)
    if not _is_complete_parquet_dir(tmp):
        raise RuntimeError(f"Spark write incomplete: {tmp}")
    for i in range(attempts):
        try:
            if os.path.exists(filepath):
                shutil.rmtree(filepath)
            os.rename(tmp, filepath)
            break
        except OSError as err:          # e.g. "Directory not empty" from a slow host filesystem
            if i == attempts - 1:
                raise RuntimeError(f"could not move {tmp} to {filepath}: {err}") from err
            time.sleep(2)
    if not _is_complete_parquet_dir(filepath):
        raise RuntimeError(f"output missing after write: {filepath}")
