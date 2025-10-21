import pandas as pd
import numpy as np
from pathlib import Path

train_path = Path(r"C:\Users\Admin\Documents\GitHub\DataMiningProj\cleaned\cleaned_train.csv")
test_path = Path(r"C:\Users\Admin\Documents\GitHub\DataMiningProj\cleaned\cleaned_test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

income_col = None
for c in train.columns:
    if "total_income" in c.lower() or "target" in c.lower():
        income_col = c
        break

if income_col is None:
    raise ValueError("Total Income column not found in train dataset.")

labels = ["Very Low", "Low", "Medium", "High", "Very High"]
vals = train[income_col].astype(float)
q = vals.quantile([0, .2, .4, .6, .8, 1.0]).values

if len(np.unique(q)) == 6:
    train["Derived_Target_Income_Status"] = pd.cut(vals, bins=q, labels=labels, include_lowest=True)
else:
    ranks = vals.rank(method="average", pct=True)
    bins = np.ceil(ranks * 5).astype(int)
    bins[bins == 0] = 1
    mapping = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
    train["Derived_Target_Income_Status"] = bins.map(mapping)

test["Derived_Target_Income_Status"] = ""

percentiles = vals.quantile([0, .2, .4, .6, .8, 1.0]).to_dict()
print("Income Percentile Ranges:", percentiles)

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

print("Derived_Target_Income_Status added to both train and test CSVs.")
print(f"Updated train shape: {train.shape}, test shape: {test.shape}")
