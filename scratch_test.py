import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
from preprocess import load_german_credit_v2, serialize_german_v2

df = load_german_credit_v2()
text_df = serialize_german_v2(df)

first_row = df.iloc[0]
text = text_df.iloc[0]["text"]

print("Tabular data:")
print(first_row.to_dict())
print("\nText:")
print(text)
