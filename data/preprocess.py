import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from splits import stratified_split, save_splits

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"


def load_german_credit() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "german_credit_risk.csv", index_col=0, keep_default_na=False)
    df["Saving accounts"] = df["Saving accounts"].replace({"NA": "Unknown", "": "Unknown"})
    df["Checking account"] = df["Checking account"].replace({"NA": "Unknown", "": "Unknown"})
    df["target"] = (df["Risk"] == "good").astype(int)
    df.drop(columns=["Risk"], inplace=True)
    return df


def load_loans() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "loans_full_schema.csv")
    terminal_statuses = ["Fully Paid", "Charged Off", "Default"]
    df = df[df["loan_status"].isin(terminal_statuses)].copy()
    df["target"] = (df["loan_status"] == "Fully Paid").astype(int)
    df.drop(columns=["loan_status"], inplace=True)
    return df


def normalize_numerics(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "target"]
    if not num_cols:
        return df
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


GERMAN_TEMPLATE = (
    "The applicant is {Sex}, aged {Age}, with job level {Job}. "
    "They live in {Housing} housing. "
    "Their saving accounts are {Saving accounts} and checking account is {Checking account}. "
    "They requested a credit amount of {Credit amount} over {Duration} months for {Purpose}."
)

LOAN_TEMPLATE = (
    "The applicant works as {emp_title} for {emp_length} years in {state}, "
    "with {homeownership} housing and an annual income of {annual_income}. "
    "Their verified income status is {verified_income} with a debt-to-income ratio of {debt_to_income}. "
    "They have {delinq_2y} delinquencies in the past 2 years, {total_credit_lines} total credit lines, "
    "and a total credit limit of {total_credit_limit}. "
    "Their account never-delinquency percentage is {account_never_delinq_percent}. "
    "They applied for a {loan_purpose} loan of {loan_amount} with a term of {term} months "
    "at {interest_rate}% interest, graded as {grade}."
)


def serialize_german(df: pd.DataFrame) -> pd.DataFrame:
    texts = df.apply(lambda row: GERMAN_TEMPLATE.format(**row), axis=1)
    return pd.DataFrame({"text": texts, "target": df["target"].values})


def serialize_loans(df: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "emp_title", "emp_length", "state", "homeownership", "annual_income",
        "verified_income", "debt_to_income", "delinq_2y", "total_credit_lines",
        "total_credit_limit", "account_never_delinq_percent", "loan_purpose",
        "loan_amount", "term", "interest_rate", "grade",
    ]
    texts = df.apply(
        lambda row: LOAN_TEMPLATE.format(**{k: row[k] for k in fields}),
        axis=1,
    )
    return pd.DataFrame({"text": texts, "target": df["target"].values})


def main():
    german = load_german_credit()
    german = normalize_numerics(german)
    german_text = serialize_german(german)
    save_splits("german", *stratified_split(german_text))

    loans = load_loans()
    loans = normalize_numerics(loans)
    loans_text = serialize_loans(loans)
    save_splits("loans", *stratified_split(loans_text))


if __name__ == "__main__":
    main()
