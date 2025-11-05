import os
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import chi2_contingency, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

path = r"C:\Users\Admin\Documents\GitHub\DataMiningProj\LTF Challenge data with dictionary.xlsx"
income_col = "Target_Variable/Total Income"

xls = pd.ExcelFile(path)
train = xls.parse(xls.sheet_names[0]).copy()
test = xls.parse(xls.sheet_names[1]).copy()

print("started")

if income_col in test.columns:
    test = test.drop(columns=[income_col])

id_like = [c for c in train.columns if any(x in c.lower() for x in ["id", "uuid", "index"])]
train = train.drop(columns=id_like, errors="ignore")
test = test.drop(columns=id_like, errors="ignore")

def clean_frame(frame):
    frame = frame.copy()
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    num_imp = SimpleImputer(strategy="median")
    cat_imp = SimpleImputer(strategy="most_frequent")
    if numeric:
        frame[numeric] = num_imp.fit_transform(frame[numeric])
    if categorical:
        frame[categorical] = cat_imp.fit_transform(frame[categorical])
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame, numeric, categorical

train_clean, train_numeric, train_categorical = clean_frame(train)
test_clean, test_numeric, test_categorical = clean_frame(test)

print("step1_loaded_counts -> train_rows:", train_clean.shape[0], "train_cols:", train_clean.shape[1],
      "test_rows:", test_clean.shape[0], "test_cols:", test_clean.shape[1])
print("step1_feature_counts -> numeric_count:", len(train_numeric), "categorical_count:", len(train_categorical))
print("step1_columns_index_preview -> train_cols_index: 1..", train_clean.shape[1], "test_cols_index: 1..", test_clean.shape[1])

if income_col not in train_clean.columns:
    raise ValueError("Income column missing")

train_clean["IncomeCategory"] = pd.qcut(
    train_clean[income_col],
    q=4,
    labels=["Low", "Medium", "High", "Very High"],
    duplicates="drop"
)
counts_by_cat = train_clean["IncomeCategory"].value_counts().reindex(["Low", "Medium", "High", "Very High"]).fillna(0).astype(int)
print("step3_income_bins ->", counts_by_cat.to_dict())

common_features = [c for c in train_clean.columns if c in test_clean.columns and c != income_col]
print("step4_common_feature_count ->", len(common_features))

x_train = train_clean[common_features].copy()
x_test = test_clean[common_features].copy()
for c in x_train.select_dtypes(include=['object', 'category']).columns:
    x_train[c] = x_train[c].astype(str)
    if c in x_test.columns:
        x_test[c] = x_test[c].astype(str)

onehot_cols = [c for c in x_train.columns if x_train[c].nunique() < 50 and x_train[c].dtype == "object"]
label_cols = [c for c in x_train.columns if x_train[c].dtype == "object" and x_train[c].nunique() >= 50]

for c in label_cols:
    le = LabelEncoder()
    if c not in x_test.columns:
        x_test[c] = ""
    full = pd.concat([x_train[c].astype(str), x_test[c].astype(str)], axis=0)
    le.fit(full)
    x_train[c] = le.transform(x_train[c].astype(str))
    x_test[c] = le.transform(x_test[c].astype(str))

x_train_dummies = pd.get_dummies(x_train[onehot_cols], drop_first=False) if onehot_cols else pd.DataFrame(index=x_train.index)
x_test_dummies = pd.get_dummies(x_test[onehot_cols], drop_first=False) if onehot_cols else pd.DataFrame(index=x_test.index)
x_train_rest = x_train.drop(columns=onehot_cols, errors="ignore")
x_test_rest = x_test.drop(columns=onehot_cols, errors="ignore")

x_train_enc = pd.concat([x_train_rest.reset_index(drop=True), x_train_dummies.reset_index(drop=True)], axis=1)
x_test_dummies_aligned = x_test_dummies.reindex(columns=x_train_dummies.columns, fill_value=0) if not x_train_dummies.empty else x_test_dummies
x_test_enc = pd.concat([x_test_rest.reset_index(drop=True), x_test_dummies_aligned.reset_index(drop=True)], axis=1)

num_cols_all = x_train_enc.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
if num_cols_all:
    x_train_enc[num_cols_all] = scaler.fit_transform(x_train_enc[num_cols_all])
    for col in num_cols_all:
        if col not in x_test_enc.columns:
            x_test_enc[col] = 0.0
    x_test_enc[num_cols_all] = scaler.transform(x_test_enc[num_cols_all])

print("step5_preprocessing_done -> final_variable_count:", x_train_enc.shape[1])

y_train = train_clean["IncomeCategory"].astype(str).reset_index(drop=True)
le = LabelEncoder()
y_enc = le.fit_transform(y_train)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}
X_tr, X_val, y_tr, y_val = train_test_split(x_train_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

metrics = {}
preds = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    p_val = model.predict(X_val)
    acc = accuracy_score(y_val, p_val)
    prec = precision_score(y_val, p_val, average="weighted", zero_division=0)
    rec = recall_score(y_val, p_val, average="weighted", zero_division=0)
    f1 = f1_score(y_val, p_val, average="weighted", zero_division=0)
    metrics[name] = {"Accuracy": round(acc, 4), "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4)}
    missing_cols = [c for c in X_tr.columns if c not in x_test_enc.columns]
    for mc in missing_cols:
        x_test_enc[mc] = 0
    x_test_enc = x_test_enc[X_tr.columns]
    preds[name] = le.inverse_transform(model.predict(x_test_enc))

metrics_df = pd.DataFrame(metrics).T
best_model = metrics_df.sort_values(by=["F1", "Accuracy"], ascending=False).index[0]
print("step6_model_metrics")
print(metrics_df)
print("step6_best_model ->", best_model)

for k, v in preds.items():
    test_clean[f"Pred_{k}"] = v
test_clean["BestModelPred"] = test_clean[f"Pred_{best_model}"]
print("step7_predictions_added -> prediction_cols:", len([c for c in test_clean.columns if c.startswith('Pred_')]), "and BestModelPred")

print("step7_predictions_sample_first5")
print(test_clean[[c for c in test_clean.columns if c.startswith('Pred_')] + ["BestModelPred"]].head(5).to_string(index=False))

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.values.sum()
    r, k = confusion_matrix.shape
    denom = min(r - 1, k - 1)
    return np.sqrt((chi2 / n) / denom) if n > 0 and denom > 0 else 0

features_for_clustering = [
    "Non_Agriculture_Income",
    "Total_Land_For_Agriculture",
    "perc_of_pop_living_in_hh_electricity",
    "Households_with_improved_Sanitation_Facility",
    "perc_Households_with_Pucca_House_That_Has_More_Than_3_Rooms",
    "Night light index",
    "Road density (Km/ SqKm)",
    "K022-Proximity to nearest mandi (Km)",
    "K022-Proximity to nearest railway (Km)",
    "Kharif Seasons  Irrigated area in 2022",
    "Rabi Seasons  Season Irrigated area in 2022",
    "Kharif Seasons  Seasonal average groundwater replenishment rate (cm) in 2022",
    "Rabi Seasons Seasonal average groundwater thickness (cm) in 2022",
    "Village score based on socio-economic parameters (0 to 100)",
    "REGION"
]

features_for_clustering = [f for f in features_for_clustering if f in x_test_enc.columns]

if not features_for_clustering:
    x_for_clustering = x_test_enc.copy()
else:
    x_for_clustering = x_test_enc[features_for_clustering].copy()

vt = VarianceThreshold(threshold=0.01)
if x_for_clustering.shape[1] > 0:
    try:
        x_vt = pd.DataFrame(vt.fit_transform(x_for_clustering),
                            columns=[c for c, v in zip(x_for_clustering.columns, vt.get_support()) if v],
                            index=x_for_clustering.index)
    except Exception:
        x_vt = x_for_clustering.copy()
else:
    x_vt = x_for_clustering.copy()

infra = [c for c in ['perc_of_pop_living_in_hh_electricity', 'Road density (Km/ SqKm)', 'Night light index'] if c in x_vt.columns]
agri = [c for c in ['Kharif Seasons  Irrigated area in 2022', 'Rabi Seasons  Season Irrigated area in 2022', 'Kharif Seasons  Seasonal average groundwater replenishment rate (cm) in 2022', 'Rabi Seasons  Seasonal average groundwater thickness (cm) in 2022'] if c in x_vt.columns]
wealth = [c for c in ['Non_Agriculture_Income', 'perc_Households_with_Pucca_House_That_Has_More_Than_3_Rooms', 'Total_Land_For_Agriculture'] if c in x_vt.columns]

indices = pd.DataFrame(index=x_vt.index)
if infra:
    indices['infra_idx'] = x_vt[infra].mean(axis=1)
if agri:
    indices['agri_idx'] = x_vt[agri].mean(axis=1)
if wealth:
    indices['wealth_idx'] = x_vt[wealth].mean(axis=1)
if indices.shape[1] == 0:
    x_compact = x_vt.copy()
else:
    sc = StandardScaler()
    x_compact = pd.DataFrame(sc.fit_transform(indices), columns=indices.columns, index=indices.index)

pca_components = min(3, x_compact.shape[1]) if x_compact.shape[1] > 0 else 1
if x_compact.shape[1] > 0:
    pca = PCA(n_components=pca_components, random_state=42)
    x_pca = pca.fit_transform(x_compact)
else:
    x_pca = x_vt.values

requested_k = 4
n_samples = x_pca.shape[0]
k_use = requested_k if n_samples >= requested_k else max(1, n_samples)

km = KMeans(n_clusters=k_use, random_state=42, n_init=10)
labels = km.fit_predict(x_pca)
test_clean["ClusterLabel"] = labels

sil = None
if k_use > 1 and x_pca.shape[0] > k_use:
    sil = silhouette_score(x_pca, labels)

ct = pd.crosstab(test_clean["ClusterLabel"], test_clean["BestModelPred"])
chi2 = chi2_contingency(ct)[0]
n = ct.values.sum()
r, k = ct.shape
cramerV = (np.sqrt((chi2 / n) / min(r - 1, k - 1))) if n > 0 and min(r - 1, k - 1) > 0 else 0

print(f"\nFORCED clustering k={requested_k} -> used_k={k_use} | silhouette: {round(sil,4) if sil is not None else 'N/A'} | Cramér's V: {round(cramerV,4)}")

counts = test_clean["ClusterLabel"].value_counts().sort_index()
print("\nCluster sizes:")
print(counts.to_string())

print("\nOne-line summaries (count, majority predicted income, top-3 numeric means):")
for cl in sorted(test_clean["ClusterLabel"].unique()):
    subset = test_clean[test_clean["ClusterLabel"] == cl]
    cnt = int(subset.shape[0])
    if "BestModelPred" in subset.columns and not subset["BestModelPred"].isnull().all():
        pred_dist = subset["BestModelPred"].value_counts(normalize=True) * 100
        top_pred = pred_dist.idxmax()
        top_pred_pct = round(pred_dist.max(), 1)
    else:
        top_pred = None
        top_pred_pct = 0.0
    numeric_cols = [c for c in features_for_clustering if c in subset.select_dtypes(include=[np.number]).columns]
    top_num_means = {}
    if numeric_cols:
        means = subset[numeric_cols].mean().sort_values(ascending=False)
        top_num_means = {k: (round(v, 2) if isinstance(v, float) else int(v)) for k, v in means.head(3).to_dict().items()}
    print(f"Cluster {cl}: n={cnt} | majority_pred={top_pred} ({top_pred_pct}%) | top_numeric_means={top_num_means}")

out_file = os.path.join(os.path.dirname(path), "final_output_compact.csv")
test_clean.to_csv(out_file, index=False)
print(f"\nSaved -> {out_file} | shape: {test_clean.shape}")
