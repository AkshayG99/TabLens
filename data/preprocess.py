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


# https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
PROPERTY_CLEANUP = {
    "if not A121/A122 : car or other, not in attribute 6": "car or other property",
    "if not A121 : building society savings agreement/ life insurance": "building society savings agreement / life insurance",
}

CHECKING_ACCOUNT_CLEANUP = {
    "... < 0 DM": "less than 0 DM",
    "0 <= ... < 200 DM": "0 to 200 DM",
    "... >= 200 DM / salary assignments for at least 1 year": "200 DM or more, or salary assignments for at least 1 year",
}

SAVINGS_ACCOUNT_CLEANUP = {
    "... < 100 DM": "less than 100 DM",
    "100 <= ... < 500 DM": "100 to 500 DM",
    "500 <= ... < 1000 DM": "500 to 1000 DM",
    ".. >= 1000 DM": "1000 DM or more",
}

EMPLOYMENT_SINCE_CLEANUP = {
    "... < 1 year": "less than 1 year",
    "1 <= ... < 4 years": "1 to 4 years",
    "4 <= ... < 7 years": "4 to 7 years",
    ".. >= 7 years": "7 years or more",
}


def load_german_credit_v2() -> pd.DataFrame:
    # german_credit_v2 is full version of UCI German Credit dataset
    df = pd.read_csv(RAW_DIR / "german_credit_uci.csv")
    df["target"] = (df["risk"] == "good").astype(int)
    df.drop(columns=["risk"], inplace=True)
    df["property"] = df["property"].replace(PROPERTY_CLEANUP)
    df["checking_account_status"] = df["checking_account_status"].replace(CHECKING_ACCOUNT_CLEANUP)
    df["savings_account_bonds"] = df["savings_account_bonds"].replace(SAVINGS_ACCOUNT_CLEANUP)
    df["present_employment_since"] = df["present_employment_since"].replace(EMPLOYMENT_SINCE_CLEANUP)
    return df


COL_RENAME = {
    "addr_state": "state",
    "home_ownership": "homeownership",
    "annual_inc": "annual_income",
    "verification_status": "verified_income",
    "dti": "debt_to_income",
    "delinq_2yrs": "delinq_2y",
    "total_acc": "total_credit_lines",
    "tot_hi_cred_lim": "total_credit_limit",
    "pct_tl_nvr_dlq": "account_never_delinq_percent",
    "purpose": "loan_purpose",
    "loan_amnt": "loan_amount",
    "int_rate": "interest_rate",
}

USED_COLUMNS = [
    "loan_status", "emp_title", "emp_length", "addr_state", "home_ownership",
    "annual_inc", "verification_status", "dti", "delinq_2yrs", "total_acc",
    "tot_hi_cred_lim", "pct_tl_nvr_dlq", "purpose", "loan_amnt", "term",
    "int_rate", "grade",
]


def load_loans() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "accepted_2007_to_2018q4.csv", usecols=USED_COLUMNS)

    terminal_statuses = ["Fully Paid", "Charged Off", "Default", "Rejected"]
    df = df[df["loan_status"].isin(terminal_statuses)].copy()

    df["target"] = (df["loan_status"] == "Fully Paid").astype(int)
    df.drop(columns=["loan_status"], inplace=True)

    df["emp_length"] = (
        df["emp_length"]
        .str.replace(r"\+?\s*years?", "", regex=True)
        .str.replace(r"<\s*", "", regex=True)
        .str.strip()
    )
    df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")

    df["term"] = df["term"].str.strip().str.replace(r"\s*months", "", regex=True)
    df["term"] = pd.to_numeric(df["term"], errors="coerce")

    df.rename(columns=COL_RENAME, inplace=True)
    df.dropna(subset=["emp_length", "debt_to_income"], inplace=True)

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

GERMAN_V2_TEMPLATE = (
    "The applicant is {personal_status_sex}, aged {age}, foreign worker: {foreign_worker}. "
    "They work as {job} and have been employed since {present_employment_since}. "
    "They live in {housing} housing, at their present residence for {present_residence_since} years, "
    "with other installment plans: {other_installment_plans}. "
    "Their property: {property}. Other debtors/guarantors: {other_debtors_guarantors}. "
    "They have {existing_credits_at_bank} existing credit(s) at this bank and are liable for "
    "maintenance of {liable_for_maintenance} dependent(s). Telephone: {telephone}. "
    "Their checking account status is {checking_account_status} and their savings account/bonds "
    "status is {savings_account_bonds}. Their credit history: {credit_history}. "
    "They requested a credit amount of {credit_amount} over {duration} months for {purpose}, "
    "with an installment rate of {installment_rate_pct_disposable_income}% of disposable income."
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


def serialize_german_v2(df: pd.DataFrame) -> pd.DataFrame:
    texts = df.apply(lambda row: GERMAN_V2_TEMPLATE.format(**row), axis=1)
    return pd.DataFrame({"text": texts, "target": df["target"].values})


def serialize_loans(df: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "emp_title", "emp_length", "state", "homeownership", "annual_income",
        "verified_income", "debt_to_income", "delinq_2y", "total_credit_lines",
        "total_credit_limit", "account_never_delinq_percent", "loan_purpose",
        "loan_amount", "term", "interest_rate", "grade",
    ]

    def _row_to_text(row):
        vals = {}
        for k in fields:
            v = row[k]
            if pd.isna(v):
                vals[k] = "Unknown"
            else:
                vals[k] = v
        return LOAN_TEMPLATE.format(**vals)

    texts = df.apply(_row_to_text, axis=1)
    return pd.DataFrame({"text": texts, "target": df["target"].values})


def main():
    german = load_german_credit()
    german = normalize_numerics(german)
    german_text = serialize_german(german)
    save_splits("german", *stratified_split(german_text))

    german_v2 = load_german_credit_v2()
    german_v2_text = serialize_german_v2(german_v2)
    save_splits("german_v2", *stratified_split(german_v2_text))

    loans = load_loans()
    loans = normalize_numerics(loans)
    loans_text = serialize_loans(loans)
    save_splits("loans", *stratified_split(loans_text))


if __name__ == "__main__":
    main()
