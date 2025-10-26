import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


train_path = Path(r"C:\Users\Admin\Documents\GitHub\DataMiningProj\cleaned\clustered_train.csv")
out_folder = train_path.parent
df = pd.read_csv(train_path, low_memory=False)
print("Step 1: Data loaded:", df.shape)


key_features = [
    "total_land_for_agriculture", "land_holding_index_source_total_agri_area_no_of_people",
    "kharif_seasons_irrigated_area_in_2022", "rabi_seasons_season_irrigated_area_in_2022",
    "k022_seasonal_average_rainfall_mm", "r022_seasonal_average_rainfall_mm",
    "road_density_km_sqkm", "night_light_index", "avg_disbursement_amount_bureau",
    "village_score_based_on_socio_economic_parameters_0_to_100", "non_agriculture_income",
    "credit_score", "avg_income_per_family", "avg_expense_per_family",
    "irrigation_facility_index", "education_index", "soil_fertility_index",
    "avg_saving_per_month", "village_development_index", "farming_experience_years"
]
features = [f for f in key_features if f in df.columns]

df = df.dropna(subset=["Cluster_Label"])
balanced_df = df.groupby("Cluster_Label", group_keys=False).apply(
    lambda x: resample(x, n_samples=min(2500, len(x)), random_state=42)
)
X = balanced_df[features]
y = balanced_df["Cluster_Label"].astype(str)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
total_samples = X_scaled.shape[0]
print("Step 2: Variables selected:", len(features), "| Balanced dataset:", X_scaled.shape)

clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, random_state=42)
clf.fit(X_scaled, y)
tree = clf.tree_
feature_names = list(X.columns)


def extract_rules(tree, feature_names, total_samples):
    rules = []
    def recurse(node, path):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            thr = tree.threshold[node]
            recurse(tree.children_left[node], path + [f"{name} <= {thr:.2f}"])
            recurse(tree.children_right[node], path + [f"{name} > {thr:.2f}"])
        else:
            vals = tree.value[node][0]
            total = np.sum(vals)
            label = np.argmax(vals)
            support = total / total_samples
            confidence = vals[label] / total
            cluster_base = np.sum(tree.value[0][0]) / total_samples
            lift = confidence / cluster_base if cluster_base > 0 else 0
            rules.append({
                "Rule": " and ".join(path),
                "Predicted_Cluster": str(label),
                "Support": round(support, 4),
                "Confidence": round(confidence, 4),
                "Lift": round(lift, 3),
                "Samples": int(total)
            })
    recurse(0, [])
    return pd.DataFrame(rules)

rules_df = extract_rules(tree, feature_names, total_samples)

rules_df["Readable_Rule"] = rules_df["Rule"].apply(lambda x: x.replace(">", " high ").replace("<=", " low ").replace(" and ", ", then "))
bins = np.linspace(rules_df["Confidence"].min(), rules_df["Confidence"].max(), 6)
labels = ["Very Low", "Low", "Medium", "High", "Very High"]
rules_df["Income_Status"] = pd.cut(rules_df["Confidence"], bins=bins, labels=labels, include_lowest=True)
rules_df = rules_df.sort_values(by=["Confidence", "Support"], ascending=False)

out_path = out_folder / "final_strong_rules.csv"
rules_df.to_csv(out_path, index=False)

summary = rules_df["Income_Status"].value_counts().reset_index()
summary.columns = ["Income_Status", "Num_Rules"]
summary.to_csv(out_folder / "final_summary.csv", index=False)

print("\nRule extraction complete.")
print("Strong rules saved to:", out_path)
print("\nSummary by Income Level:\n", summary)
