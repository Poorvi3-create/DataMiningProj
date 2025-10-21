import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

train_path = Path(r"C:\Users\Admin\Documents\GitHub\DataMiningProj\cleaned\cleaned_train.csv")
test_path = Path(r"C:\Users\Admin\Documents\GitHub\DataMiningProj\cleaned\cleaned_test.csv")
out_folder = train_path.parent

train = pd.read_csv(train_path, low_memory=False)
test = pd.read_csv(test_path, low_memory=False)

exclude_keywords = ["derived_target", "income", "target", "id", "farmerid"]
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if not any(k in c.lower() for k in exclude_keywords)]
numeric_cols = [c for c in numeric_cols if train[c].nunique() > 1]

train_num = train[numeric_cols].copy()
test_num = test[numeric_cols].copy() if not test.empty else None

train_num = train_num.fillna(train_num.median())
if test_num is not None:
    test_num = test_num.fillna(train_num.median())

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_num)
test_scaled = scaler.transform(test_num) if test_num is not None else None

n_features = train_scaled.shape[1]
n_components = min(10, n_features)
pca = PCA(n_components=n_components, random_state=42)
train_pca = pca.fit_transform(train_scaled)
test_pca = pca.transform(test_scaled) if test_scaled is not None else None

ks = range(2, 11)
wcss = []
sil_scores = []
sample_size = min(5000, train_pca.shape[0])
rng = np.random.default_rng(42)

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(train_pca)
    wcss.append(km.inertia_)
    if train_pca.shape[0] > sample_size:
        idx = rng.choice(train_pca.shape[0], size=sample_size, replace=False)
        sil = silhouette_score(train_pca[idx], labels[idx])
    else:
        sil = silhouette_score(train_pca, labels)
    sil_scores.append(sil)

best_k = ks[int(np.argmax(sil_scores))]
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
train_clusters = km_final.fit_predict(train_pca)
train["Cluster_Label"] = train_clusters
if test_pca is not None:
    test_clusters = km_final.predict(test_pca)
    test["Cluster_Label"] = test_clusters

centroids_pca = km_final.cluster_centers_
centroids_scaled = pca.inverse_transform(centroids_pca)
centroids_original = scaler.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(centroids_original, columns=numeric_cols)
centroid_df.index.name = "cluster"
cluster_sizes = train["Cluster_Label"].value_counts().sort_index().rename("size")

(train.assign(_tmp=0).drop(columns=["_tmp"], errors="ignore")
 .to_csv(out_folder / "cleaned_train_clustered.csv", index=False))
if not test.empty:
    test.to_csv(out_folder / "cleaned_test_clustered.csv", index=False)
centroid_df.to_csv(out_folder / "cluster_centroids.csv")
cluster_sizes.to_frame().to_csv(out_folder / "cluster_sizes.csv")

print("selected_k:", best_k)
print("wcss:", dict(zip(ks, wcss)))
print("silhouette_scores:", dict(zip(ks, sil_scores)))
print("cluster_sizes:\n", cluster_sizes)
print("cluster_centroids saved to:", out_folder / "cluster_centroids.csv")
print("clustered train saved to:", out_folder / "cleaned_train_clustered.csv")
if not test.empty:
    print("clustered test saved to:", out_folder / "cleaned_test_clustered.csv")
