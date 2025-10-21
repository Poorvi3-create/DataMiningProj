
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.metrics import silhouette_score

TRAIN = Path(r"C:\Users\Admin\Documents\GitHub\DataMiningProj\cleaned\cleaned_train.csv")
TEST  = Path(r"C:\Users\Admin\Documents\GitHub\DataMiningProj\cleaned\cleaned_test.csv")
OUT   = TRAIN.parent

FEATURES = [
    "total_land_for_agriculture",
    "land_holding_index_source_total_agri_area_no_of_people",
    "kharif_seasons_irrigated_area_in_2022",
    "rabi_seasons_season_irrigated_area_in_2022",
    "rabi_seasons_seasonal_average_groundwater_thickness_cm_in_2022",
    "k022_seasonal_average_rainfall_mm",
    "r022_seasonal_average_rainfall_mm",
    "r021_seasonal_average_rainfall_mm",
    "kharif_seasons_agricultural_score_in_2022",
    "rabi_seasons_agricultural_score_in_2022",
    "road_density_km_sqkm",
    "night_light_index",
    "village_score_based_on_socio_economic_parameters_0_to_100",
    "kharif_seasons_cropping_density_in_2022",
    "non_agriculture_income",
    "avg_disbursement_amount_bureau"
]

def safe_silhouette(X, labels, preferred=500, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = X.shape[0]
    if n <= preferred:
        try:
            return float(silhouette_score(X, labels))
        except Exception:
            return -1.0
    # try use sample
    size = min(preferred, n)
    for attempt in (size, max(200, size//2), 100, 50):
        try:
            idx = rng.choice(n, size=attempt, replace=False)
            return float(silhouette_score(X[idx], labels[idx]))
        except MemoryError:
            continue
        except Exception:
            return -1.0
    return -1.0

def main():
    train = pd.read_csv(TRAIN, low_memory=False)
    test = pd.read_csv(TEST, low_memory=False)

    present = [f for f in FEATURES if f in train.columns]
    if len(present) == 0:
        raise SystemExit("None of the selected features found in train CSV.")

    train_mat = train[present].fillna(train[present].median())
    test_mat = test[present].fillna(train[present].median()) if not test.empty else None

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_mat)
    test_s = scaler.transform(test_mat) if test_mat is not None else None

    n_comp = min(3, train_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    train_pca = pca.fit_transform(train_s)
    test_pca = pca.transform(test_s) if test_s is not None else None

    ks = range(2, 9)
    wcss = []
    sils = []
    rng = np.random.default_rng(42)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=1, algorithm="full")
        labels = km.fit_predict(train_pca)
        wcss.append(km.inertia_)
        sils.append(safe_silhouette(train_pca, labels, preferred=500, rng=rng))

    best_idx = int(np.argmax(sils))
    best_k = list(ks)[best_idx]
    if all(s <= -0.5 for s in sils):  # fallback if silhouette failed for all
        best_k = 5

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    train_labels = km_final.fit_predict(train_pca)
    test_labels = km_final.predict(test_pca) if test_pca is not None else None
    train["Cluster_Label"] = train_labels
    if test_pca is not None:
        test["Cluster_Label"] = test_labels

    centroids_pca = km_final.cluster_centers_
    centroids_scaled = pca.inverse_transform(centroids_pca)
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids_orig, columns=present)
    centroid_df.index.name = "Cluster"

    overall = train[present].mean()

    def one_line(row):
        diff = (row - overall).sort_values(ascending=False)
        pos = diff.head(2).index.tolist()
        neg = diff.tail(1).index.tolist()
        parts = []
        if any("irrigat" in s for s in pos): parts.append("high irrigation")
        if any("land" in s or "area" in s or "holding" in s for s in pos): parts.append("large land")
        if any("night" in s or "light" in s for s in pos): parts.append("high infra")
        if any("non_agriculture" in s for s in pos): parts.append("non-agri income")
        if any("rain" in s for s in neg): parts.append("low rainfall")
        if not parts:
            parts = ["mixed profile"]
        return (", ".join(parts).capitalize() + " farmers")

    centroid_df["OneLineDescription"] = centroid_df.apply(one_line, axis=1)
    counts = train["Cluster_Label"].value_counts().sort_index().rename("Farmer_Count")

    train.to_csv(OUT / "clustered_train.csv", index=False)
    if not test.empty:
        test.to_csv(OUT / "clustered_test.csv", index=False)
    centroid_df.to_csv(OUT / "cluster_centroids.csv")
    counts.to_frame().to_csv(OUT / "cluster_counts.csv")

    print("done")
    print("best_k:", best_k)
    print("wcss:", dict(zip(list(ks), wcss)))
    print("sils:", dict(zip(list(ks), sils)))
    print("counts:\n", counts)

if __name__ == "__main__":
    main()
