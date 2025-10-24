import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Step 1: Loading datasets...")
train_path = Path(r"C:\Users\Admin\Documents\GitHub\DataMiningProj\cleaned\cleaned_train.csv")
test_path = Path(r"C:\Users\Admin\Documents\GitHub\DataMiningProj\cleaned\cleaned_test.csv")
out_folder = train_path.parent
train = pd.read_csv(train_path, low_memory=False)
test = pd.read_csv(test_path, low_memory=False)
print("Train shape:", train.shape, "| Test shape:", test.shape)

selected_features = [
    "total_land_for_agriculture",
    "land_holding_index_source_total_agri_area_no_of_people",
    "kharif_seasons_irrigated_area_in_2022",
    "rabi_seasons_season_irrigated_area_in_2022",
    "rabi_seasons_seasonal_average_groundwater_thickness_cm_in_2022",
    "k022_seasonal_average_rainfall_mm",
    "r022_seasonal_average_rainfall_mm",
    "night_light_index",
    "road_density_km_sqkm",
    "village_score_based_on_socio_economic_parameters_0_to_100",
    "avg_disbursement_amount_bureau",
    "no_of_active_loan_in_bureau",
    "non_agriculture_income",
    "households_with_improved_sanitation_facility",
    "perc_of_pop_living_in_hh_electricity",
    "perc_households_with_pucca_house_that_has_more_than_3_rooms",
    "perc_of_wall_material_with_burnt_brick",
    "mat_roof_metal_gi_asbestos_sheets",
    "kharif_seasons_agricultural_score_in_2022",
    "rabi_seasons_agricultural_score_in_2022",
    "kharif_seasons_cropping_density_in_2022",
    "rabi_seasons_cropping_density_in_2022",
    "k022_village_category_based_on_agri_parameters_good_average_poor",
    "r022_village_category_based_on_agri_parameters_good_average_poor",
    "rabi_seasons_agro_ecological_sub_zone_in_2022"
]

print("Step 2: Selecting and cleaning variables...")
available_features = [f for f in selected_features if f in train.columns]
train_num = train[available_features].copy()
test_num = test[available_features].copy()
print("Variables used:", len(available_features))

categorical_cols = train_num.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_num[col], test_num[col]], axis=0).astype(str)
    le.fit(combined)
    train_num[col] = le.transform(train_num[col].astype(str))
    test_num[col] = le.transform(test_num[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded categorical column: {col}")

train_num = train_num.fillna(train_num.median(numeric_only=True))
test_num = test_num.fillna(train_num.median(numeric_only=True))

print("Step 3: Scaling and reducing dimensions...")
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_num)
test_scaled = scaler.transform(test_num)
pca = PCA(n_components=4, random_state=42)
train_pca = pca.fit_transform(train_scaled)
test_pca = pca.transform(test_scaled)
print("PCA complete. Explained variance ratio:", np.sum(pca.explained_variance_ratio_).round(3))

print("Step 4: Finding optimal number of clusters...")
ks = range(2, 10)
sil_scores = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels = km.fit_predict(train_pca)
    sil = silhouette_score(train_pca, labels)
    sil_scores.append(sil)
    print(f"K={k}, Silhouette={sil:.3f}")
best_k = ks[int(np.argmax(sil_scores))]
print("Optimal K:", best_k, "| Best Silhouette:", max(sil_scores))

print("Step 5: Final clustering with optimal K...")
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
train["Cluster_Label"] = km_final.fit_predict(train_pca)
test["Cluster_Label"] = km_final.predict(test_pca)
print("Cluster assignment complete.")

print("Step 6: Calculating centroids and counts...")
centroids_pca = km_final.cluster_centers_
centroids_scaled = pca.inverse_transform(centroids_pca)
centroids_original = scaler.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(centroids_original, columns=available_features)
centroid_df.index.name = "Cluster"
centroid_df["Count"] = train["Cluster_Label"].value_counts().sort_index().values

print("Step 7: Saving results...")
train.to_csv(out_folder / "clustered_train.csv", index=False)
test.to_csv(out_folder / "clustered_test.csv", index=False)
centroid_df.to_csv(out_folder / "cluster_centroids.csv")

print("\nClustering complete.")
print("Cluster centroids saved to:", out_folder / "cluster_centroids.csv")
print("Clustered train/test data saved with 'Cluster_Label'.")
print("\nCluster Sizes:\n", train["Cluster_Label"].value_counts().sort_index())
