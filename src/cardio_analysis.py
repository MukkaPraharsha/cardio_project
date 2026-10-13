# %% [markdown]
# # Cardiovascular Disease Prediction
# End-to-end pipeline: preprocessing -> EDA/visualisation -> correlation matrix ->
# SVM / KNN / Decision Tree / Logistic Regression / Random Forest -> best model saved.

# %%
import os, json, time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report)
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, MOD = f"{ROOT}/figures", f"{ROOT}/models"
RS = 42
def save(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=110); plt.close()

# %% [markdown]
# ## 1. Load data
# %%
df = pd.read_csv(f"{ROOT}/data/cardio_train.csv", sep=";")
print("Shape:", df.shape); print(df.head()); df.info()
print(df.describe().T)
print("Missing values:\n", df.isnull().sum())
print("Duplicates (excluding id):", df.drop(columns="id").duplicated().sum())

# %% [markdown]
# ## 2. Pre-processing
# - drop `id`, convert age (days -> years), remove duplicates
# - remove physiologically impossible values (BP, height, weight)
# - feature engineering: BMI, pulse pressure, MAP
# %%
raw_n = len(df)
df = df.drop(columns="id").drop_duplicates()
df["age"] = (df["age"] / 365.25).round().astype(int)
df["gender"] = df["gender"].map({1: 0, 2: 1})        # 0 = female, 1 = male (dataset convention: 1=women, 2=men)
n0 = len(df)
df = df[(df.ap_hi.between(70, 250)) & (df.ap_lo.between(40, 190)) & (df.ap_hi > df.ap_lo)]
df = df[df.height.between(120, 220) & df.weight.between(30, 200)]
df = df.reset_index(drop=True)
print(f"Rows: raw={raw_n}, after dedupe={n0}, after outlier removal={len(df)}")
df["bmi"] = df.weight / (df.height / 100) ** 2
df = df[df.bmi.between(12, 60)].reset_index(drop=True)
df["pulse_pressure"] = df.ap_hi - df.ap_lo
df["map"] = (df.ap_hi + 2 * df.ap_lo) / 3
print("Final rows:", len(df))
df.to_csv(f"{ROOT}/data/cardio_clean.csv", index=False)

# %% [markdown]
# ## 3. Exploratory data analysis
# %%
lab = {0: "No disease", 1: "Disease"}
plt.figure(figsize=(5, 4)); ax = sns.countplot(x=df.cardio.map(lab))
for c in ax.containers: ax.bar_label(c)
plt.title("Target distribution"); save("01_target_distribution")

num = ["age", "height", "weight", "ap_hi", "ap_lo", "bmi"]
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for a, c in zip(axs.ravel(), num):
    sns.histplot(df[c], kde=True, ax=a); a.set_title(f"Distribution of {c}")
save("02_numeric_distributions")

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for a, c in zip(axs.ravel(), num):
    sns.boxplot(x=df.cardio.map(lab), y=df[c], ax=a); a.set_title(f"{c} vs cardio"); a.set_xlabel("")
save("03_boxplots_by_target")

cats = {"gender": "Gender (0=F,1=M)", "cholesterol": "Cholesterol (1 normal,2 above,3 well above)",
        "gluc": "Glucose (1 normal,2 above,3 well above)", "smoke": "Smoker", "alco": "Alcohol", "active": "Physically active"}
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for a, (c, t) in zip(axs.ravel(), cats.items()):
    sns.countplot(x=c, hue=df.cardio.map(lab), data=df, ax=a); a.set_title(t, fontsize=10); a.legend(title="")
save("04_categorical_vs_target")

fig, axs = plt.subplots(2, 3, figsize=(16, 8))
for a, c in zip(axs.ravel(), cats):
    r = df.groupby(c).cardio.mean() * 100
    sns.barplot(x=r.index, y=r.values, ax=a); a.set_ylabel("% with disease"); a.set_title(f"Disease rate by {c}")
    for p in a.patches: a.annotate(f"{p.get_height():.1f}%", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
save("05_disease_rate_by_category")

df["age_group"] = pd.cut(df.age, [29, 40, 45, 50, 55, 60, 70], labels=["<=40", "41-45", "46-50", "51-55", "56-60", "60+"])
df["bmi_class"] = pd.cut(df.bmi, [0, 18.5, 25, 30, 35, 100], labels=["Under", "Normal", "Over", "Obese I", "Obese II+"])
def bp_cat(r):
    if r.ap_hi >= 140 or r.ap_lo >= 90: return "Hypertension"
    if r.ap_hi >= 120 or r.ap_lo >= 80: return "Elevated"
    return "Normal"
df["bp_cat"] = df.apply(bp_cat, axis=1)
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
for a, c in zip(axs, ["age_group", "bmi_class", "bp_cat"]):
    r = df.groupby(c, observed=True).cardio.mean() * 100
    sns.barplot(x=r.index.astype(str), y=r.values, ax=a); a.set_title(f"Disease rate by {c}"); a.set_ylabel("%")
    for p in a.patches: a.annotate(f"{p.get_height():.0f}%", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
save("06_risk_by_age_bmi_bp")
print(df.groupby("age_group", observed=True).cardio.mean().round(3)); print(df.groupby("bp_cat").cardio.mean().round(3)); print(df.groupby("bmi_class", observed=True).cardio.mean().round(3))

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
s = df.sample(6000, random_state=RS)
sns.scatterplot(x="ap_lo", y="ap_hi", hue=s.cardio.map(lab), data=s, alpha=.5, ax=axs[0]); axs[0].set_title("Systolic vs diastolic BP")
sns.kdeplot(data=df, x="age", hue=df.cardio.map(lab), fill=True, ax=axs[1]); axs[1].set_title("Age density by outcome")
save("07_bp_scatter_age_kde")

g = sns.pairplot(df.sample(2000, random_state=RS)[["age", "bmi", "ap_hi", "ap_lo", "cardio"]], hue="cardio", corner=True, plot_kws={"alpha": .4, "s": 12})
g.figure.savefig(f"{FIG}/08_pairplot.png", dpi=90); plt.close("all")

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(df.pivot_table(index="cholesterol", columns="gluc", values="cardio", aggfunc="mean") * 100, annot=True, fmt=".0f", cmap="Reds", ax=axs[0])
axs[0].set_title("% disease: cholesterol x glucose")
sns.heatmap(df.pivot_table(index="age_group", columns="bp_cat", values="cardio", aggfunc="mean", observed=True) * 100, annot=True, fmt=".0f", cmap="Reds", ax=axs[1])
axs[1].set_title("% disease: age group x BP category")
save("09_interaction_heatmaps")

# %% [markdown]
# ## 4. Correlation matrix
# %%
model_cols = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "bmi", "pulse_pressure", "map"]
corr = df[model_cols + ["cardio"]].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, mask=np.triu(np.ones_like(corr, dtype=bool), 1), square=True)
plt.title("Correlation matrix"); save("10_correlation_matrix")
plt.figure(figsize=(6, 6)); corr["cardio"].drop("cardio").sort_values().plot.barh(color="teal")
plt.title("Correlation with cardio"); save("11_correlation_with_target")
print(corr["cardio"].sort_values(ascending=False).round(3))

# %% [markdown]
# ## 5. Modelling: SVM, KNN, Decision Tree, Logistic Regression, Random Forest
# %%
features = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "bmi"]
X, y = df[features], df.cardio
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
print("Train/test:", Xtr.shape, Xte.shape)

models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000, random_state=RS))]),
    "KNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=25))]),
    "SVM (RBF)": Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", C=1.0, random_state=RS))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=RS),
}
rows, fitted, scores = [], {}, {}
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
for name, m in models.items():
    t = time.time()
    if name == "SVM (RBF)":   # CV on a 10k subsample: SVM training is O(n^2)
        idx = np.random.RandomState(RS).choice(len(Xtr), 10000, replace=False)
        cvs = cross_val_score(m, Xtr.iloc[idx], ytr.iloc[idx], cv=cv, scoring="accuracy").mean()
    else:
        cvs = cross_val_score(m, Xtr, ytr, cv=cv, scoring="accuracy").mean()
    if name == "SVM (RBF)":   # SVM trained on a 20k stratified subsample (kernel SVM scales ~O(n^2))
        sub = np.random.RandomState(RS).choice(len(Xtr), 20000, replace=False)
        m.fit(Xtr.iloc[sub], ytr.iloc[sub])
    else:
        m.fit(Xtr, ytr)
    p = m.predict(Xte)
    sc = m.decision_function(Xte) if hasattr(m, "decision_function") else m.predict_proba(Xte)[:, 1]
    fitted[name], scores[name] = m, sc
    rows.append(dict(Model=name, CV_Accuracy=cvs, Test_Accuracy=accuracy_score(yte, p), Precision=precision_score(yte, p),
                     Recall=recall_score(yte, p), F1=f1_score(yte, p), ROC_AUC=roc_auc_score(yte, sc), Seconds=time.time() - t))
    print(f"{name:20s} acc={rows[-1]['Test_Accuracy']:.4f} auc={rows[-1]['ROC_AUC']:.4f} ({rows[-1]['Seconds']:.0f}s)", flush=True)
res = pd.DataFrame(rows).sort_values("Test_Accuracy", ascending=False).reset_index(drop=True)
res.round(4).to_csv(f"{MOD}/model_comparison.csv", index=False)
print(res.round(4).to_string())

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
rm = res.melt(id_vars="Model", value_vars=["Test_Accuracy", "Precision", "Recall", "F1"])
sns.barplot(x="Model", y="value", hue="variable", data=rm, ax=axs[0]); axs[0].set_ylim(.6, .8); axs[0].tick_params(axis="x", rotation=20); axs[0].set_title("Model metrics")
for n, s in scores.items():
    f, t, _ = roc_curve(yte, s); axs[1].plot(f, t, label=f"{n} (AUC={roc_auc_score(yte, s):.3f})")
axs[1].plot([0, 1], [0, 1], "k--"); axs[1].legend(); axs[1].set_title("ROC curves"); axs[1].set_xlabel("FPR"); axs[1].set_ylabel("TPR")
save("12_model_comparison_roc")
fig, axs = plt.subplots(1, 5, figsize=(24, 4.5))
for a, (n, m) in zip(axs, fitted.items()):
    ConfusionMatrixDisplay.from_predictions(yte, m.predict(Xte), ax=a, cmap="Blues", colorbar=False); a.set_title(n)
save("13_confusion_matrices")

# %% [markdown]
# ## 6. Tune the strongest candidate and build the final model
# %%
param = {"n_estimators": [200, 300, 500], "max_depth": [8, 10, 12, 14], "min_samples_leaf": [3, 5, 10, 20], "max_features": ["sqrt", 0.5]}
rs = RandomizedSearchCV(RandomForestClassifier(n_jobs=-1, random_state=RS), param, n_iter=8, cv=3, scoring="accuracy", random_state=RS)
rs.fit(Xtr, ytr)
print("Best RF params:", rs.best_params_, "CV acc:", round(rs.best_score_, 4))
tuned = rs.best_estimator_
tuned_acc = accuracy_score(yte, tuned.predict(Xte))
print(f"Tuned RF test acc={tuned_acc:.4f}  AUC={roc_auc_score(yte, tuned.predict_proba(Xte)[:,1]):.4f}")

best_name, best_model, best_acc = res.Model[0], fitted[res.Model[0]], res.Test_Accuracy[0]
if tuned_acc >= best_acc:
    best_name, best_model, best_acc = "Random Forest (tuned)", tuned, tuned_acc
bp = best_model.predict(Xte)
print("FINAL MODEL:", best_name, round(best_acc, 4)); print(classification_report(yte, bp, target_names=["No disease", "Disease"]))
ConfusionMatrixDisplay.from_predictions(yte, bp, display_labels=["No disease", "Disease"], cmap="Blues"); plt.title(f"Final: {best_name}"); save("14_final_confusion_matrix")

imp = pd.Series(tuned.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 6)); imp.plot.barh(color="darkred"); plt.title("Random Forest feature importance"); save("15_feature_importance")
print(imp.sort_values(ascending=False).round(4))

joblib.dump({"model": best_model, "features": features, "name": best_name}, f"{MOD}/best_model.joblib")
json.dump({"final_model": best_name, "test_accuracy": round(best_acc, 4), "rf_best_params": rs.best_params_,
           "n_rows_clean": len(df), "features": features}, open(f"{MOD}/summary.json", "w"), indent=2)
print("Saved model.")
