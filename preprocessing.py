import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys

FILE_PATH = r"C:\Users\Admin\Documents\GitHub\DataMiningProj\LTF Challenge data with dictionary.xlsx"
OUT_FOLDER = "cleaned"
MISSING_THRESH = 0.5

def normalize_name(name):
    s = str(name).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[-/()%,]", "_", s)
    s = re.sub(r"__+", "_", s)
    s = s.strip("_").lower()
    return s

def try_convert_numeric(series):
    s = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() / max(1, len(series)) > 0.2:
        return s_num
    return series

def drop_high_missing(df, thresh=MISSING_THRESH):
    keep = df.columns[df.isna().mean() <= thresh]
    dropped = [c for c in df.columns if c not in set(keep)]
    return df[keep], dropped

def fill_missing(df):
    nums = df.select_dtypes(include=[np.number]).columns
    cats = df.select_dtypes(include=["object", "category"]).columns
    for c in nums:
        df[c] = df[c].fillna(df[c].median())
    for c in cats:
        mode = df[c].mode()
        df[c] = df[c].fillna(mode[0] if not mode.empty else "Unknown")
    return df

def convert_categories(df, unique_thresh=50):
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() <= unique_thresh:
            df[c] = df[c].astype("category")
    return df

def clean_frame(df, drop_candidates, missing_thresh=MISSING_THRESH):
    df = df.copy()
    mapping = {c: normalize_name(c) for c in df.columns}
    df = df.rename(columns=mapping)
    normalized_drop = [normalize_name(x) for x in drop_candidates]
    present_to_drop = [c for c in df.columns if c in normalized_drop]
    if present_to_drop:
        df = df.drop(columns=present_to_drop, errors="ignore")
        print("Dropped identifiers:", present_to_drop)
    for col in list(df.select_dtypes(include=["object"]).columns):
        df[col] = try_convert_numeric(df[col])
    df, dropped_due_missing = drop_high_missing(df, thresh=missing_thresh)
    if dropped_due_missing:
        print("Dropped sparse columns:", dropped_due_missing)
    df = fill_missing(df)
    df = convert_categories(df)
    return df

def clean_both(file_path=FILE_PATH, train_sheet_name=None, test_sheet_name=None, out_folder=OUT_FOLDER):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    xls = pd.ExcelFile(p)
    sheets = xls.sheet_names
    train_name = train_sheet_name if train_sheet_name and train_sheet_name in sheets else ("TrainData" if "TrainData" in sheets else sheets[0])
    test_name = test_sheet_name if test_sheet_name and test_sheet_name in sheets else (None if len(sheets) < 2 else ("TestData" if "TestData" in sheets else sheets[1]))
    train_df = pd.read_excel(xls, sheet_name=train_name)
    test_df = pd.read_excel(xls, sheet_name=test_name) if test_name else None
    print("Loaded", p.name, "sheets:", sheets)
    print("Using train sheet:", train_name, " test sheet:", test_name)
    drop_candidates = [
    "FarmerID",
    "Village",
    "Zipcode",
    "CITY",
    "Location",
    "Nearest Mandi name - Kharif 22",
    "mandi from nearest railwaystation - Kharif 22",
    "Loan Agreement no",
    "Address type",
    "Label"
    ]
    cleaned_train = clean_frame(train_df, drop_candidates, missing_thresh=MISSING_THRESH)
    cleaned_test = clean_frame(test_df, drop_candidates, missing_thresh=MISSING_THRESH) if test_df is not None else None
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    train_out = Path(out_folder) / "cleaned_train.csv"
    cleaned_train.to_csv(train_out, index=False)
    print("Saved cleaned train to:", train_out)
    if cleaned_test is not None:
        test_out = Path(out_folder) / "cleaned_test.csv"
        cleaned_test.to_csv(test_out, index=False)
        print("Saved cleaned test to:", test_out)
    print("Final shapes -> train:", cleaned_train.shape, " test:", cleaned_test.shape if cleaned_test is not None else None)
    return cleaned_train, cleaned_test

if __name__ == "__main__":
    cleaned_train, cleaned_test = clean_both()
