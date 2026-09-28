import os
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read_bronze(snapshot_date_str, bronze_directory, table_name, spark):
    partition_name = f"bronze_{table_name}_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = os.path.join(bronze_directory, partition_name)
    # bronze is stored as raw strings; silver enforces the schema explicitly below
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print("loaded from:", filepath, "row count:", df.count())
    return df


def _write_silver(df, snapshot_date_str, silver_directory, table_name):
    partition_name = f"silver_{table_name}_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = os.path.join(silver_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)


def _clean_numeric(column_name, dtype=FloatType()):
    """Strip stray underscores / whitespace (e.g. '52312.68_', '__10000__') and cast.
    Values that still cannot be parsed (e.g. '_') become null."""
    cleaned = F.trim(F.regexp_replace(col(column_name), "_", ""))
    cleaned = F.when(cleaned == "", None).otherwise(cleaned)
    # cast via double first so that strings like '11.0' survive an integer cast
    return cleaned.cast("double").cast(dtype)


def _null_outside(column_name, low, high):
    """Domain rule: values outside a plausible business range are data-entry errors -> null.
    Bounds are FIXED business rules, not statistics computed from the data, so no
    information from the future / test period leaks into the pipeline."""
    return F.when((col(column_name) >= low) & (col(column_name) <= high), col(column_name)).otherwise(None)


# ---------------------------------------------------------------------------
# loan management system (label source) - from Lab 2
# ---------------------------------------------------------------------------
def process_silver_loan_daily(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    df = _read_bronze(snapshot_date_str, bronze_lms_directory, "loan_daily", spark)

    # clean data: enforce schema / data type
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    _write_silver(df, snapshot_date_str, silver_loan_daily_directory, "loan_daily")
    return df


# ---------------------------------------------------------------------------
# customer attributes
# ---------------------------------------------------------------------------
def process_silver_attributes(snapshot_date_str, bronze_directory, silver_directory, spark):
    df = _read_bronze(snapshot_date_str, bronze_directory, "attributes", spark)

    # Name and SSN are kept in silver: silver is the cleaned single source of truth that other
    # consumers (KYC, fraud, collections) rely on. They are PII, so they are removed at gold.
    # Name: trim whitespace only. Corrupted spellings (e.g. 'ODonnell"d') are left as-is rather than guessed.
    df = df.withColumn("Name", F.trim(col("Name")))
    df = df.withColumn("Name", F.when(col("Name") != "", col("Name")))
    # SSN: keep only well-formed 'ddd-dd-dddd' values; garbage such as '#F%$D@*&8' -> null + flag
    df = df.withColumn("ssn_valid", F.coalesce(F.trim(col("SSN")).rlike(r"^\d{3}-\d{2}-\d{4}$"), F.lit(False)).cast(IntegerType()))
    df = df.withColumn("SSN", F.when(col("ssn_valid") == 1, F.trim(col("SSN"))))

    # Age: strip underscores ('32_'), cast, and null out impossible ages (-500, 8678, ...).
    # Genuine ages in the source run 14-56, so the lower bound keeps the 14-17 year olds.
    df = df.withColumn("Age", _clean_numeric("Age", IntegerType()))
    df = df.withColumn("Age", _null_outside("Age", 14, 100))

    # Occupation: placeholder '_______' -> null
    df = df.withColumn("Occupation", F.when(col("Occupation").rlike("[A-Za-z]"), col("Occupation")).otherwise(None))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    df = df.select("Customer_ID", "Name", "SSN", "ssn_valid", "Age", "Occupation", "snapshot_date")
    _write_silver(df, snapshot_date_str, silver_directory, "attributes")
    return df


def check_silver_attributes_quality(silver_directory, spark):
    """Data-quality check across all attribute partitions: a valid SSN should belong to
    exactly one Customer_ID. Prints a warning rather than failing the pipeline."""
    import glob
    paths = glob.glob(os.path.join(silver_directory, "silver_attributes_*.parquet"))
    if not paths:
        return 0
    df = spark.read.parquet(*paths).filter(col("ssn_valid") == 1)
    shared = df.groupBy("SSN").agg(F.countDistinct("Customer_ID").alias("n")).filter(col("n") > 1).count()
    invalid = spark.read.parquet(*paths).filter(col("ssn_valid") == 0).count()
    print(f"[silver:attributes DQ] invalid SSNs: {invalid} | SSNs shared by >1 Customer_ID: {shared}")
    if shared > 0:
        print("WARNING: some SSNs map to more than one Customer_ID - check for duplicate customers")
    return shared


# ---------------------------------------------------------------------------
# customer financials
# ---------------------------------------------------------------------------
FINANCIAL_NUMERIC_COLS = {
    "Annual_Income": FloatType(),
    "Monthly_Inhand_Salary": FloatType(),
    "Num_Bank_Accounts": IntegerType(),
    "Num_Credit_Card": IntegerType(),
    "Interest_Rate": IntegerType(),
    "Num_of_Loan": IntegerType(),
    "Delay_from_due_date": IntegerType(),
    "Num_of_Delayed_Payment": IntegerType(),
    "Changed_Credit_Limit": FloatType(),
    "Num_Credit_Inquiries": IntegerType(),
    "Outstanding_Debt": FloatType(),
    "Credit_Utilization_Ratio": FloatType(),
    "Total_EMI_per_month": FloatType(),
    "Amount_invested_monthly": FloatType(),
    "Monthly_Balance": FloatType(),
}

# fixed business-rule ranges; anything outside is treated as a data-entry error
FINANCIAL_VALID_RANGES = {
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 50),          # % p.a.
    "Num_of_Loan": (0, 20),            # removes -100 and 1,495
    "Num_of_Delayed_Payment": (0, 50),
    "Num_Credit_Inquiries": (0, 50),
    "Monthly_Balance": (-1e6, 1e6),    # removes the -3.3e26 sentinel
}

PAYMENT_BEHAVIOUR_VALUES = [
    "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments",
    "High_spent_Small_value_payments", "High_spent_Medium_value_payments", "High_spent_Large_value_payments",
]


def process_silver_financials(snapshot_date_str, bronze_directory, silver_directory, spark):
    df = _read_bronze(snapshot_date_str, bronze_directory, "financials", spark)

    # 1. numeric columns: strip underscores and enforce types
    for column, dtype in FINANCIAL_NUMERIC_COLS.items():
        df = df.withColumn(column, _clean_numeric(column, dtype))

    # 2. out-of-range values -> null
    for column, (low, high) in FINANCIAL_VALID_RANGES.items():
        df = df.withColumn(column, _null_outside(column, low, high))

    # Annual income should be roughly 12x monthly in-hand salary; inflated values (up to 23.8M) -> null
    df = df.withColumn("Annual_Income", F.when(col("Annual_Income") <= 3 * 12 * col("Monthly_Inhand_Salary"), col("Annual_Income")))
    # EMI larger than the whole monthly salary is not plausible -> null
    df = df.withColumn("Total_EMI_per_month", F.when(col("Total_EMI_per_month") <= col("Monthly_Inhand_Salary"), col("Total_EMI_per_month")))
    # '__10000__' is a sentinel value, not a real investment amount -> null
    df = df.withColumn("Amount_invested_monthly", F.when(col("Amount_invested_monthly") != 10000, col("Amount_invested_monthly")))

    # 3. Credit_History_Age '10 Years and 9 Months' -> 129 months
    years = F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s*Years", 1)
    months = F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s*Months", 1)
    df = df.withColumn(
        "Credit_History_Age_months",
        F.when(years != "", years.cast(IntegerType()) * 12 + F.coalesce(months.cast(IntegerType()), F.lit(0))).cast(IntegerType()),
    ).drop("Credit_History_Age")

    # 4. categorical placeholders -> null
    df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix").isin("Good", "Standard", "Bad"), col("Credit_Mix")))
    df = df.withColumn("Payment_of_Min_Amount", F.when(col("Payment_of_Min_Amount").isin("Yes", "No", "NM"), col("Payment_of_Min_Amount")))
    df = df.withColumn("Payment_Behaviour", F.when(col("Payment_Behaviour").isin(*PAYMENT_BEHAVIOUR_VALUES), col("Payment_Behaviour")))

    # 5. Type_of_Loan: 'Auto Loan, and Payday Loan' -> 'Auto Loan,Payday Loan' (null = no loans)
    df = df.withColumn("Type_of_Loan", F.regexp_replace(col("Type_of_Loan"), r"\s*,\s*(and\s+)?", ","))
    df = df.withColumn("Type_of_Loan", F.regexp_replace(col("Type_of_Loan"), r"^and\s+", ""))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    _write_silver(df, snapshot_date_str, silver_directory, "financials")
    return df


# ---------------------------------------------------------------------------
# clickstream
# ---------------------------------------------------------------------------
CLICKSTREAM_COLS = [f"fe_{i}" for i in range(1, 21)]


def process_silver_clickstream(snapshot_date_str, bronze_directory, silver_directory, spark):
    df = _read_bronze(snapshot_date_str, bronze_directory, "clickstream", spark)

    for c in CLICKSTREAM_COLS:
        df = df.withColumn(c, _clean_numeric(c, IntegerType()))
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # one row per customer per month
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])

    _write_silver(df, snapshot_date_str, silver_directory, "clickstream")
    return df
