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
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import textwrap

warnings.filterwarnings("ignore")

path = r"C:\Users\Admin\Documents\GitHub\DataMiningProj\LTF Challenge data with dictionary.xlsx"
incomecol = "Target_Variable/Total Income"

xls = pd.ExcelFile(path)
traindata = xls.parse(xls.sheet_names[0]).copy()
testdata = xls.parse(xls.sheet_names[1]).copy()

print("Started")

if incomecol in testdata.columns:
    testdata = testdata.drop(columns=[incomecol])

idcols = [c for c in traindata.columns if any(x in c.lower() for x in ["id", "uuid", "index"])]
traindata = traindata.drop(columns=idcols, errors="ignore")
testdata = testdata.drop(columns=idcols, errors="ignore")

def cleanframe(frame):
    frame = frame.copy()
    numcols = frame.select_dtypes(include=[np.number]).columns.tolist()
    catcols = [c for c in frame.columns if c not in numcols]
    numimp = SimpleImputer(strategy="median")
    catimp = SimpleImputer(strategy="most_frequent")
    if numcols:
        frame[numcols] = numimp.fit_transform(frame[numcols])
    if catcols:
        frame[catcols] = catimp.fit_transform(frame[catcols])
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame, numcols, catcols

trainclean, trainnumcols, traincatcols = cleanframe(traindata)
testclean, testnumcols, testcatcols = cleanframe(testdata)

print("Step1 Loaded Counts = Train Rows:", trainclean.shape[0], "Train Cols:", trainclean.shape[1], "Test Rows:", testclean.shape[0], "Test Cols:", testclean.shape[1])
print("Step1 Feature Counts = Numeric Count:", len(trainnumcols), "Categorical Count:", len(traincatcols))
print("Step1 Columns Index Preview = Train Cols Index: 1..", trainclean.shape[1], "Test Cols Index: 1..", testclean.shape[1])

if incomecol not in trainclean.columns:
    raise ValueError("Income column missing")

trainclean["incomecat"] = pd.qcut(trainclean[incomecol], q=4, labels=["Low", "Medium", "High", "Very High"], duplicates="drop")
countinc = trainclean["incomecat"].value_counts().reindex(["Low", "Medium", "High", "Very High"]).fillna(0).astype(int)
print("Step3 Income Bins =", countinc.to_dict())

commoncols = [c for c in trainclean.columns if c in testclean.columns and c != incomecol]
print("Step4 Common Feature Count =", len(commoncols))

xtrain = trainclean[commoncols].copy()
xtest = testclean[commoncols].copy()
for c in xtrain.select_dtypes(include=['object', 'category']).columns:
    xtrain[c] = xtrain[c].astype(str)
    if c in xtest.columns:
        xtest[c] = xtest[c].astype(str)

onehotcols = [c for c in xtrain.columns if xtrain[c].nunique() < 50 and xtrain[c].dtype == "object"]
labelcols = [c for c in xtrain.columns if xtrain[c].dtype == "object" and xtrain[c].nunique() >= 50]

for c in labelcols:
    le = LabelEncoder()
    if c not in xtest.columns:
        xtest[c] = ""
    full = pd.concat([xtrain[c].astype(str), xtest[c].astype(str)], axis=0)
    le.fit(full)
    xtrain[c] = le.transform(xtrain[c].astype(str))
    xtest[c] = le.transform(xtest[c].astype(str))

xtraindum = pd.get_dummies(xtrain[onehotcols], drop_first=False) if onehotcols else pd.DataFrame(index=xtrain.index)
xtestdum = pd.get_dummies(xtest[onehotcols], drop_first=False) if onehotcols else pd.DataFrame(index=xtest.index)
xtrainrest = xtrain.drop(columns=onehotcols, errors="ignore")
xtestrest = xtest.drop(columns=onehotcols, errors="ignore")

xtrainenc = pd.concat([xtrainrest.reset_index(drop=True), xtraindum.reset_index(drop=True)], axis=1)
xtestdumalign = xtestdum.reindex(columns=xtraindum.columns, fill_value=0) if not xtraindum.empty else xtestdum
xtestenc = pd.concat([xtestrest.reset_index(drop=True), xtestdumalign.reset_index(drop=True)], axis=1)

numall = xtrainenc.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
if numall:
    xtrainenc[numall] = scaler.fit_transform(xtrainenc[numall])
    for col in numall:
        if col not in xtestenc.columns:
            xtestenc[col] = 0.0
    xtestenc[numall] = scaler.transform(xtestenc[numall])

print("Step5 Preprocessing Done = Final Variable Count:", xtrainenc.shape[1])

ytrain = trainclean["incomecat"].astype(str).reset_index(drop=True)
le = LabelEncoder()
yenc = le.fit_transform(ytrain)

models = {
    "randomforest": RandomForestClassifier(n_estimators=100, random_state=42),
    "decisiontree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier()
}
xtr, xval, ytr, yval = train_test_split(xtrainenc, yenc, test_size=0.2, random_state=42, stratify=yenc)

metrics = {}
preds = {}
for name, model in models.items():
    model.fit(xtr, ytr)
    pval = model.predict(xval)
    acc = accuracy_score(yval, pval)
    prec = precision_score(yval, pval, average="weighted", zero_division=0)
    rec = recall_score(yval, pval, average="weighted", zero_division=0)
    f1 = f1_score(yval, pval, average="weighted", zero_division=0)
    metrics[name] = {"Accuracy": round(acc, 4), "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4)}
    missingcols = [c for c in xtr.columns if c not in xtestenc.columns]
    for mc in missingcols:
        xtestenc[mc] = 0
    xtestenc = xtestenc[xtr.columns]
    preds[name] = le.inverse_transform(model.predict(xtestenc))

metricsdf = pd.DataFrame(metrics).T
bestmodel = metricsdf.sort_values(by=["F1", "Accuracy"], ascending=False).index[0]
print("Step6 Model Metrics")
print(metricsdf)
print("Step6 Best Model =", bestmodel)

for k, v in preds.items():
    testclean[f"pred{k}"] = v
testclean["bestmodelpred"] = testclean[f"pred{bestmodel}"]
print("Step7 Predictions Added = Prediction Cols:", len([c for c in testclean.columns if c.startswith('pred')]), "And BestModelPred")

print("Step7 Predictions Sample First5")
print(testclean[[c for c in testclean.columns if c.startswith('pred')] + ["bestmodelpred"]].head(5).to_string(index=False))

featuresforclust = [
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

featuresforclust = [f for f in featuresforclust if f in xtestenc.columns]

if not featuresforclust:
    xforclust = xtestenc.copy()
else:
    xforclust = xtestenc[featuresforclust].copy()

vt = VarianceThreshold(threshold=0.01)
if xforclust.shape[1] > 0:
    try:
        xvt = pd.DataFrame(vt.fit_transform(xforclust), columns=[c for c, v in zip(xforclust.columns, vt.get_support()) if v], index=xforclust.index)
    except Exception:
        xvt = xforclust.copy()
else:
    xvt = xforclust.copy()

infra = [c for c in ['perc_of_pop_living_in_hh_electricity', 'Road density (Km/ SqKm)', 'Night light index'] if c in xvt.columns]
agri = [c for c in ['Kharif Seasons  Irrigated area in 2022', 'Rabi Seasons  Season Irrigated area in 2022', 'Kharif Seasons  Seasonal average groundwater replenishment rate (cm) in 2022', 'Rabi Seasons  Seasonal average groundwater thickness (cm) in 2022'] if c in xvt.columns]
wealth = [c for c in ['Non_Agriculture_Income', 'perc_Households_with_Pucca_House_That_Has_More_Than_3_Rooms', 'Total_Land_For_Agriculture'] if c in xvt.columns]

indices = pd.DataFrame(index=xvt.index)
if infra:
    indices['infraidx'] = xvt[infra].mean(axis=1)
if agri:
    indices['agriidx'] = xvt[agri].mean(axis=1)
if wealth:
    indices['wealthidx'] = xvt[wealth].mean(axis=1)
if indices.shape[1] == 0:
    xcompact = xvt.copy()
else:
    sc = StandardScaler()
    xcompact = pd.DataFrame(sc.fit_transform(indices), columns=indices.columns, index=indices.index)

pcacomp = min(3, xcompact.shape[1]) if xcompact.shape[1] > 0 else 1
if xcompact.shape[1] > 0:
    pca = PCA(n_components=pcacomp, random_state=42)
    xpca = pca.fit_transform(xcompact)
else:
    xpca = xvt.values

nsamp = xpca.shape[0]
mink = 2
maxk = 10
valid_ks = [k for k in range(mink, maxk + 1) if nsamp > k]

sil_scores = {}
for k in range(mink, maxk + 1):
    if nsamp <= k:
        sil_scores[k] = None
        continue
    try:
        kmtemp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = kmtemp.fit_predict(xpca)
        sil = silhouette_score(xpca, labs)
        sil_scores[k] = float(sil)
    except Exception:
        sil_scores[k] = None

print("\nSilhouette Scores By K")
for k in range(mink, maxk + 1):
    sc = sil_scores.get(k)
    print(f"K={k} | Silhouette: {round(sc,4) if sc is not None else 'N/A'}")

valid_scores = {k: v for k, v in sil_scores.items() if v is not None}
if valid_scores:
    best_k = max(valid_scores.items(), key=lambda x: x[1])[0]
    best_score = valid_scores[best_k]
else:
    best_k = 1
    best_score = None

if best_k > 1:
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(xpca)
else:
    labels = np.zeros(nsamp, dtype=int)

testclean["clusterlabel"] = labels

sil_final = None
if best_k > 1 and nsamp > best_k:
    try:
        sil_final = silhouette_score(xpca, labels)
    except Exception:
        sil_final = None

ct = pd.crosstab(testclean["clusterlabel"], testclean["bestmodelpred"])
chi2 = chi2_contingency(ct)[0]
n = ct.values.sum()
r, k = ct.shape
cramv = (np.sqrt((chi2 / n) / min(r - 1, k - 1))) if n > 0 and min(r - 1, k - 1) > 0 else 0

print(f"\nSelected K = {best_k} | Best Silhouette = {round(best_score,4) if best_score is not None else 'N/A'} | Final Silhouette: {round(sil_final,4) if sil_final is not None else 'N/A'} | Cramér's V: {round(cramv,4)}")

counts = testclean["clusterlabel"].value_counts().sort_index()
incomeorder = ["Low", "Medium", "High", "Very High"]
rows = []
for cl in counts.index:
    subset = testclean[testclean["clusterlabel"] == cl]
    distpct = subset["bestmodelpred"].value_counts(normalize=True).reindex(incomeorder, fill_value=0) * 100
    rows.append([counts[cl]] + [round(distpct[label], 1) for label in incomeorder])
clustertable = pd.DataFrame(rows, index=counts.index, columns=["count"] + [s.lower() for s in incomeorder])
clustertable.index.name = "clusterlabel"
displaytable = clustertable.copy()
displaytable.columns = ["Count", "Low %", "Medium %", "High %", "Very High %"]
print("\nCluster Sizes And Income Distribution (%)")
print(displaytable.to_string())

def safemean(df, cols):
    vals = [df[c].astype(float).mean(skipna=True) for c in cols if c in df.columns]
    return np.mean(vals) if vals else 0

colsinfra = ['perc_of_pop_living_in_hh_electricity','Night light index','Road density (Km/ SqKm)']
colsagri = ['Kharif Seasons  Irrigated area in 2022','Rabi Seasons  Season Irrigated area in 2022']
colswealth = ['Non_Agriculture_Income','Total_Land_For_Agriculture']
colsvillage = ['Village score based on socio-economic parameters (0 to 100)']
colsprox = ['K022-Proximity to nearest mandi (Km)','K022-Proximity to nearest railway (Km)']

print("\nCluster Insights (One-Line Summary):")
clustersummary = []
for cl in clustertable.index:
    subset = testclean[testclean["clusterlabel"] == cl]
    vals = {
        "infra": safemean(subset, colsinfra),
        "agri": safemean(subset, colsagri),
        "wealth": safemean(subset, colswealth),
        "village": safemean(subset, colsvillage),
        "prox": safemean(subset, colsprox)
    }
    clustersummary.append((cl, vals))

dfsum = pd.DataFrame([v for _, v in clustersummary], index=[c for c, _ in clustersummary])
rankings = {col: dfsum[col].rank(ascending=False, method="dense") for col in dfsum.columns}

for cl in dfsum.index:
    rankdata = {k: rankings[k][cl] for k in rankings}
    phrases = []

    if rankdata["wealth"] == 1:
        phrases.append("the most affluent cluster with the highest non-farm income and land value")
    elif rankdata["wealth"] == len(dfsum):
        phrases.append("the least affluent cluster with the lowest income and limited assets")
    elif rankdata["wealth"] <= 2:
        phrases.append("among the higher income clusters showing strong financial capacity")
    elif rankdata["wealth"] >= len(dfsum) - 1:
        phrases.append("among the lower income groups with weaker economic base")

    if rankdata["infra"] == 1:
        phrases.append("showing best infrastructure access and connectivity")
    elif rankdata["infra"] == len(dfsum):
        phrases.append("having the weakest infrastructure and poor road density")
    elif rankdata["infra"] <= 2:
        phrases.append("well connected but with uneven service access")
    else:
        phrases.append("moderate infrastructure conditions")

    if rankdata["agri"] == 1:
        phrases.append("with the highest irrigation coverage and most active agriculture")
    elif rankdata["agri"] == len(dfsum):
        phrases.append("with the least agricultural engagement and irrigation use")
    elif rankdata["agri"] <= 2:
        phrases.append("agriculture-focused with good irrigation presence")
    else:
        phrases.append("balanced agricultural activity")

    if rankdata["prox"] == 1:
        phrases.append("closest to markets and transport hubs")
    elif rankdata["prox"] == len(dfsum):
        phrases.append("most geographically remote cluster")
    else:
        phrases.append("moderately connected to trade networks")

    if rankdata["village"] == 1:
        phrases.append("socially advanced with strong village development indicators")
    elif rankdata["village"] == len(dfsum):
        phrases.append("socially lagging on basic amenities and literacy")

    desc = ", ".join(phrases[:4])
    wrapped = "\n     ".join(textwrap.wrap(desc.capitalize() + ".", width=110))
    print(f"Cluster {cl}: {wrapped}")

outfile = os.path.join(os.path.dirname(path), "final_output_compact.csv")
testclean.to_csv(outfile, index=False)
print(f"\nSaved = {outfile} | Shape = {testclean.shape}")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

outdir = os.path.dirname(path) or "."

metricsfile = os.path.join(outdir, "metrics_grouped_bar.png")
clusterfile = os.path.join(outdir, "clusters_stacked_income.png")
silhouettefile = os.path.join(outdir, "silhouette_vs_k.png")
heatmapfile = os.path.join(outdir, "corr_heatmap_top_features_clean.png")

metricswork = metricsdf.copy().astype(float)
metricsordered = metricswork.sort_values(by="F1", ascending=False)
modellist = metricsordered.index.tolist()
metricnames = ["Accuracy", "Precision", "Recall", "F1"]
metricvals = [metricsordered[m].values for m in metricnames]
xpos = np.arange(len(modellist))
barw = 0.18
plt.figure(figsize=(10, 6))
for i, vals in enumerate(metricvals):
    plt.bar(xpos + (i - 1.5) * barw, vals, width=barw, label=metricnames[i])
plt.xticks(xpos, modellist)
plt.xlabel("Model")
plt.ylabel("Score (Weighted)")
plt.title("Model Performance (Grouped) = Ordered By F1 (Desc)")
plt.ylim(0, 1.05)
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig(metricsfile, dpi=300)
plt.close()
print(f"Saved Grouped Metrics Bar Chart = {metricsfile}")

displaytablelocal = displaytable.copy()
if "Low %" in displaytablelocal.columns:
    perccolnames = ["Low %", "Medium %", "High %", "Very High %"]
else:
    perccolnames = [c for c in displaytablelocal.columns if "%" in c]
    if not perccolnames:
        temp = clustertable.copy()
        temp.columns = ["count"] + [s.lower() for s in ["Low", "Medium", "High", "Very High"]]
        displaytablelocal = temp.rename(columns={s.lower(): s.title() + " %" for s in ["Low", "Medium", "High", "Very High"]})
        perccolnames = ["Low %", "Medium %", "High %", "Very High %"]
clusterids = displaytablelocal.index.astype(str).tolist()
stackdata = displaytablelocal[perccolnames].values
plt.figure(figsize=(10, 6))
bottom = np.zeros(len(clusterids))
colors = ["#FF8A65", "#FF7043", "#FF5722", "#D84315"]
for i, col in enumerate(perccolnames):
    vals = stackdata[:, i]
    plt.bar(clusterids, vals, bottom=bottom, label=col, color=colors[i % len(colors)])
    bottom += vals
plt.xlabel("Cluster")
plt.ylabel("Percentage (%)")
plt.title("Cluster Composition By Income Category (Stacked %)")
plt.legend(title="Income Category", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(clusterfile, dpi=300)
plt.close()
print(f"Saved Cluster Income Stacked Chart = {clusterfile}")

silscores = sil_scores.copy()
klist = sorted([k for k in silscores.keys()])
silvals = [silscores.get(k) if silscores.get(k) is not None else np.nan for k in klist]
plt.figure(figsize=(9, 5))
plt.plot(klist, silvals, marker="o", linewidth=2)
plt.xticks(klist)
plt.xlabel("K (Number Of Clusters)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Vs K")
if not all(np.isnan(silvals)):
    validvals = [v for v in silvals if not np.isnan(v)]
    ymin = max(0, min(validvals) - 0.03)
    ymax = min(1, max(validvals) + 0.03)
    plt.ylim(ymin, ymax)
    plt.gca().set_ylim(ymin, ymax)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(silhouettefile, dpi=300)
plt.close()
print(f"Saved Silhouette Vs K Line Chart = {silhouettefile}")

xtrain = xtrainenc.copy()
targetseries = pd.Series(yenc, name="Target")
if len(xtrain) > len(targetseries):
    xtrain = xtrain.iloc[: len(targetseries)].reset_index(drop=True)
elif len(targetseries) > len(xtrain):
    targetseries = targetseries.iloc[: len(xtrain)].reset_index(drop=True)
corrwithtarget = xtrain.corrwith(targetseries).sort_values(key=lambda x: abs(x), ascending=False)
topn = min(10, corrwithtarget.shape[0])
topfeatures = corrwithtarget.head(topn)
corrdf = pd.DataFrame({"Feature": topfeatures.index, "Correlation": topfeatures.values})
plt.figure(figsize=(8, 6))
sns.heatmap(
    corrdf.set_index("Feature").T,
    annot=True,
    fmt=".2f",
    cmap="OrRd",
    cbar_kws={"label": "Pearson R"},
    linewidths=0.5
)
plt.title("Top 10 Feature Correlations With Target")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(heatmapfile, dpi=300)
plt.close()
print(f"Saved Correlation Heatmap = {heatmapfile}")
