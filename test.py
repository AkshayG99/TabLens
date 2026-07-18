import pandas as pd
from ucimlrepo import fetch_ucirepo

statlog_german_credit_data = fetch_ucirepo(id=144)

df = statlog_german_credit_data.data.features
df["class"] = statlog_german_credit_data.data.targets

COLUMN_MAP = {
    "Attribute1": "checking_account_status",
    "Attribute2": "duration",
    "Attribute3": "credit_history",
    "Attribute4": "purpose",
    "Attribute5": "credit_amount",
    "Attribute6": "savings_account_bonds",
    "Attribute7": "present_employment_since",
    "Attribute8": "installment_rate_pct_disposable_income",
    "Attribute9": "personal_status_sex",
    "Attribute10": "other_debtors_guarantors",
    "Attribute11": "present_residence_since",
    "Attribute12": "property",
    "Attribute13": "age",
    "Attribute14": "other_installment_plans",
    "Attribute15": "housing",
    "Attribute16": "existing_credits_at_bank",
    "Attribute17": "job",
    "Attribute18": "liable_for_maintenance",
    "Attribute19": "telephone",
    "Attribute20": "foreign_worker",
    "class": "risk",
}

CHECKING_ACCOUNT = {
    "A11": "... < 0 DM",
    "A12": "0 <= ... < 200 DM",
    "A13": "... >= 200 DM / salary assignments for at least 1 year",
    "A14": "no checking account",
}

CREDIT_HISTORY = {
    "A30": "no credits taken/ all credits paid back duly",
    "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly till now",
    "A33": "delay in paying off in the past",
    "A34": "critical account/ other credits existing (not at this bank)",
}

PURPOSE = {
    "A40": "car (new)",
    "A41": "car (used)",
    "A42": "furniture/equipment",
    "A43": "radio/television",
    "A44": "domestic appliances",
    "A45": "repairs",
    "A46": "education",
    "A48": "retraining",
    "A49": "business",
    "A410": "others",
}

SAVINGS_ACCOUNT = {
    "A61": "... < 100 DM",
    "A62": "100 <= ... < 500 DM",
    "A63": "500 <= ... < 1000 DM",
    "A64": ".. >= 1000 DM",
    "A65": "unknown/ no savings account",
}

EMPLOYMENT_SINCE = {
    "A71": "unemployed",
    "A72": "... < 1 year",
    "A73": "1 <= ... < 4 years",
    "A74": "4 <= ... < 7 years",
    "A75": ".. >= 7 years",
}

PERSONAL_STATUS_SEX = {
    "A91": "male : divorced/separated",
    "A92": "female : divorced/separated/married",
    "A93": "male : single",
    "A94": "male : married/widowed",
    "A95": "female : single",
}

OTHER_DEBTORS = {
    "A101": "none",
    "A102": "co-applicant",
    "A103": "guarantor",
}

PROPERTY = {
    "A121": "real estate",
    "A122": "if not A121 : building society savings agreement/ life insurance",
    "A123": "if not A121/A122 : car or other, not in attribute 6",
    "A124": "unknown / no property",
}

OTHER_INSTALLMENT_PLANS = {
    "A141": "bank",
    "A142": "stores",
    "A143": "none",
}

HOUSING = {
    "A151": "rent",
    "A152": "own",
    "A153": "for free",
}

JOB = {
    "A171": "unemployed/ unskilled - non-resident",
    "A172": "unskilled - resident",
    "A173": "skilled employee / official",
    "A174": "management/ self-employed/ highly qualified employee/ officer",
}

TELEPHONE = {
    "A191": "none",
    "A192": "yes, registered under the customers name",
}

FOREIGN_WORKER = {
    "A201": "yes",
    "A202": "no",
}

RISK = {
    1: "good",
    2: "bad",
}

VALUE_MAPS = {
    "checking_account_status": CHECKING_ACCOUNT,
    "credit_history": CREDIT_HISTORY,
    "purpose": PURPOSE,
    "savings_account_bonds": SAVINGS_ACCOUNT,
    "present_employment_since": EMPLOYMENT_SINCE,
    "personal_status_sex": PERSONAL_STATUS_SEX,
    "other_debtors_guarantors": OTHER_DEBTORS,
    "property": PROPERTY,
    "other_installment_plans": OTHER_INSTALLMENT_PLANS,
    "housing": HOUSING,
    "job": JOB,
    "telephone": TELEPHONE,
    "foreign_worker": FOREIGN_WORKER,
    "risk": RISK,
}

df = df.rename(columns=COLUMN_MAP)

for col, mapping in VALUE_MAPS.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

df.to_csv("data/raw/german_credit_uci.csv", index=False)
print(f"Saved {df.shape[0]} rows, {df.shape[1]} columns to data/raw/german_credit_uci.csv")
print("\n=== SAMPLE ===")
print(df.head().to_string())
