import os
import glob
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def _silver_path(silver_directory, table_name, date_str):
    return os.path.join(silver_directory, f"silver_{table_name}_" + date_str.replace("-", "_") + ".parquet")


def _write_gold(df, snapshot_date_str, gold_directory, table_name):
    partition_name = f"gold_{table_name}_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = os.path.join(gold_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)


# ---------------------------------------------------------------------------
# LABEL STORE (from Lab 2)
# ---------------------------------------------------------------------------
def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    # connect to silver table
    filepath = _silver_path(silver_loan_daily_directory, "loan_daily", snapshot_date_str)
    df = spark.read.parquet(filepath)
    print("loaded from:", filepath, "row count:", df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd) + "dpd_" + str(mob) + "mob").cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    _write_gold(df, snapshot_date_str, gold_label_store_directory, "label_store")
    return df


# ---------------------------------------------------------------------------
# FEATURE STORE
# ---------------------------------------------------------------------------
# Fixed vocabularies for encoding. They are business-defined category lists (not learnt from
# the data), so encoding is deterministic and identical for train, test and production.
OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur", "Journalist",
    "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
]
LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan", "Mortgage Loan",
    "Payday Loan", "Personal Loan", "Student Loan", "Not Specified",
]
CREDIT_MIX_ORDINAL = {"Bad": 0, "Standard": 1, "Good": 2}
PAYMENT_VALUE_ORDINAL = {"Small": 0, "Medium": 1, "Large": 2}
CLICKSTREAM_COLS = [f"fe_{i}" for i in range(1, 21)]


def _slug(s):
    return s.lower().replace("-", "_").replace(" ", "_")


def _build_clickstream_features(snapshot_date_str, silver_clickstream_directory, spark):
    """
    Point-in-time clickstream features for an application month.

    Only months <= the application month are read, so the model never sees behaviour that
    happens after the loan decision (prevents temporal leakage).

    Each monthly value is mostly noise around a stable per-customer level (month-to-month
    correlation ~0, no trend over time), so the best summary is the average over ALL months
    available up to the application date:
      - fe_i_avg                    : mean of fe_i over every month <= application month
      - clickstream_months_observed : how many months that average is based on (its reliability)
    """
    all_paths = glob.glob(os.path.join(silver_clickstream_directory, "silver_clickstream_*.parquet"))
    cutoff = snapshot_date_str.replace("-", "_")
    # partition names end in YYYY_MM_DD, so string comparison == date comparison
    paths = [p for p in all_paths if os.path.basename(p)[len("silver_clickstream_"):-len(".parquet")] <= cutoff]
    if not paths:
        return None

    cs = spark.read.parquet(*paths).filter(col("snapshot_date") <= F.lit(snapshot_date_str).cast(DateType()))
    return cs.groupBy("Customer_ID").agg(
        *[F.avg(c).cast(FloatType()).alias(f"{c}_avg") for c in CLICKSTREAM_COLS],
        F.countDistinct("snapshot_date").cast(IntegerType()).alias("clickstream_months_observed"),
    )


def process_features_gold_table(snapshot_date_str, silver_loan_daily_directory, silver_attributes_directory,
                                silver_financials_directory, silver_clickstream_directory,
                                gold_feature_store_directory, spark):
    """
    One row per loan application (snapshot_date = application date = loan start date).
    Join to the label store on loan_id:
        label_store.loan_id == feature_store.loan_id
    (the label itself is observed LABEL_MOB months later: label.snapshot_date = add_months(snapshot_date, mob))
    """
    # --- base population: loans starting this month (month-on-book 0) = the applications to score ---
    # loan terms are fixed at application, so they are valid application-time features
    loans = spark.read.parquet(_silver_path(silver_loan_daily_directory, "loan_daily", snapshot_date_str)) \
        .filter(col("mob") == 0) \
        .select("loan_id", "Customer_ID", col("loan_start_date").alias("snapshot_date"),
                col("loan_amt").cast(FloatType()), col("tenure").cast(IntegerType()))
    print(f"[gold:feature_store] {snapshot_date_str} applications: {loans.count()}")

    # --- customer attributes & financials as of the application date ---
    # Name / SSN are PII kept in silver for other consumers; they never enter the feature store
    attr = spark.read.parquet(_silver_path(silver_attributes_directory, "attributes", snapshot_date_str)) \
        .drop("Name", "SSN", "ssn_valid", "snapshot_date")
    fin = spark.read.parquet(_silver_path(silver_financials_directory, "financials", snapshot_date_str)).drop("snapshot_date")

    df = loans.join(attr, "Customer_ID", "left").join(fin, "Customer_ID", "left")

    # --- demographic encodings ---
    for occ in OCCUPATIONS:
        df = df.withColumn(f"occupation_{_slug(occ)}", (col("Occupation") == occ).cast(IntegerType()))
    df = df.withColumn("occupation_unknown", col("Occupation").isNull().cast(IntegerType()))
    for occ in OCCUPATIONS:  # unknown occupation -> 0 in every dummy (rather than null)
        df = df.withColumn(f"occupation_{_slug(occ)}", F.coalesce(col(f"occupation_{_slug(occ)}"), F.lit(0)))

    # --- credit-product encodings ---
    for lt in LOAN_TYPES:
        df = df.withColumn(f"has_{_slug(lt)}",
                           F.coalesce(F.array_contains(F.split(col("Type_of_Loan"), ","), lt), F.lit(False)).cast(IntegerType()))
    df = df.withColumn("num_loan_types", sum(col(f"has_{_slug(lt)}") for lt in LOAN_TYPES).cast(IntegerType()))

    credit_mix_expr = F.lit(None).cast(IntegerType())
    for k, v in CREDIT_MIX_ORDINAL.items():
        credit_mix_expr = F.when(col("Credit_Mix") == k, F.lit(v)).otherwise(credit_mix_expr)
    df = df.withColumn("credit_mix_ordinal", credit_mix_expr.cast(IntegerType()))

    for v in ["Yes", "No", "NM"]:
        df = df.withColumn(f"min_payment_{v.lower()}", F.coalesce((col("Payment_of_Min_Amount") == v).cast(IntegerType()), F.lit(0)))

    # Payment_Behaviour 'High_spent_Medium_value_payments' -> spend level + payment size
    df = df.withColumn("spend_level_high",
                       F.when(col("Payment_Behaviour").startswith("High"), 1).when(col("Payment_Behaviour").startswith("Low"), 0).cast(IntegerType()))
    pay_size = F.regexp_extract(col("Payment_Behaviour"), r"spent_(\w+?)_value", 1)
    pay_expr = F.lit(None).cast(IntegerType())
    for k, v in PAYMENT_VALUE_ORDINAL.items():
        pay_expr = F.when(pay_size == k, F.lit(v)).otherwise(pay_expr)
    df = df.withColumn("payment_value_ordinal", pay_expr.cast(IntegerType()))

    # --- engineered affordability ratios ---
    df = df.withColumn("debt_to_annual_income", (col("Outstanding_Debt") / col("Annual_Income")).cast(FloatType()))
    df = df.withColumn("emi_to_monthly_salary", (col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")).cast(FloatType()))
    df = df.withColumn("invested_to_monthly_salary", (col("Amount_invested_monthly") / col("Monthly_Inhand_Salary")).cast(FloatType()))
    df = df.withColumn("balance_to_monthly_salary", (col("Monthly_Balance") / col("Monthly_Inhand_Salary")).cast(FloatType()))
    df = df.withColumn("delayed_payments_per_loan",
                       F.when(col("Num_of_Loan") > 0, col("Num_of_Delayed_Payment") / col("Num_of_Loan")).cast(FloatType()))

    # --- clickstream (point-in-time, all months up to the application date) ---
    cs = _build_clickstream_features(snapshot_date_str, silver_clickstream_directory, spark)
    cs_feature_cols = [f"{c}_avg" for c in CLICKSTREAM_COLS]
    if cs is not None:
        df = df.join(cs, "Customer_ID", "left")
    else:  # no clickstream available (keeps a stable schema across partitions)
        for c in cs_feature_cols:
            df = df.withColumn(c, F.lit(None).cast(FloatType()))
        df = df.withColumn("clickstream_months_observed", F.lit(None).cast(IntegerType()))
    df = df.withColumn("clickstream_months_observed", F.coalesce(col("clickstream_months_observed"), F.lit(0)).cast(IntegerType()))
    df = df.withColumn("has_clickstream", (col("clickstream_months_observed") > 0).cast(IntegerType()))
    for c in cs_feature_cols:
        df = df.withColumn(c, col(c).cast(FloatType()))

    # --- final column selection (drop raw categoricals now that they are encoded) ---
    loan_term_cols = ["loan_amt", "tenure"]
    numeric_cols = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age_months",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    ]
    engineered_cols = [
        "debt_to_annual_income", "emi_to_monthly_salary", "invested_to_monthly_salary",
        "balance_to_monthly_salary", "delayed_payments_per_loan",
    ]
    encoded_cols = (
        [f"occupation_{_slug(o)}" for o in OCCUPATIONS] + ["occupation_unknown"]
        + [f"has_{_slug(lt)}" for lt in LOAN_TYPES] + ["num_loan_types", "credit_mix_ordinal",
           "min_payment_yes", "min_payment_no", "min_payment_nm", "spend_level_high", "payment_value_ordinal"]
    )
    clickstream_cols = ["has_clickstream", "clickstream_months_observed"] + cs_feature_cols

    df = df.select("loan_id", "Customer_ID", "snapshot_date", *loan_term_cols, *numeric_cols, *engineered_cols, *encoded_cols, *clickstream_cols)

    _write_gold(df, snapshot_date_str, gold_feature_store_directory, "feature_store")
    return df
