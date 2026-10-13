# Cardiovascular Disease Prediction

Machine-learning project that predicts whether a patient has cardiovascular disease from routine medical
examination data (age, blood pressure, cholesterol, glucose, BMI, lifestyle habits).

Heart disease is among the leading causes of death in India and worldwide (about 17.6 million deaths in 2016). Early,
reliable screening matters, so this project analyses a large patient dataset and compares five classic ML algorithms.

## Dataset
`data/cardio_train.csv` - 70,000 patient records, 11 features + target (extracted from the embedded file in the project brief).

| Feature | Description |
|---|---|
| age | days in the raw file (converted to years) |
| gender | 1 = female, 2 = male (recoded to 0/1) |
| height (cm), weight (kg) | body measurements |
| ap_hi / ap_lo | systolic / diastolic blood pressure |
| cholesterol, gluc | 1 normal, 2 above normal, 3 well above normal |
| smoke, alco, active | smoker, alcohol intake, physical activity (0/1) |
| **cardio** | **target**: 1 = cardiovascular disease, 0 = none |

## Project structure
```
cardio_project/
├── data/                 cardio_train.csv (raw), cardio_clean.csv (processed)
├── figures/              15 plots (EDA, correlation, model comparison, feature importance)
├── models/               best_model.joblib, model_comparison.csv, summary.json
├── src/cardio_analysis.py   full pipeline (preprocessing -> EDA -> models -> save)
├── src/predict.py           example prediction for a new patient
├── Cardiovascular_Disease_Prediction.ipynb   same pipeline as notebook (run from project root)
├── requirements.txt
└── run_log.txt           console output of the run that produced the results below
```

## How to run
```bash
pip install -r requirements.txt
python src/cardio_analysis.py     # ~5 min on 1 CPU; regenerates figures/ and models/
python src/predict.py             # predict for a sample patient
```

## Methodology (maps to the project instructions)

### 1. Data pre-processing
- No missing values; 24 duplicate rows removed.
- Age converted from days to years; gender recoded to 0/1; `id` dropped.
- Removed impossible values: systolic outside 70-250, diastolic outside 40-190, systolic <= diastolic,
  height outside 120-220 cm, weight outside 30-200 kg, BMI outside 12-60.
  **70,000 -> 68,566 rows (2.0% removed)**.
- Feature engineering: **BMI**, pulse pressure, mean arterial pressure (used in EDA/correlation; BMI used in models).
- Stratified 80/20 train/test split; standard scaling inside pipelines for LR, KNN, SVM (no leakage).

### 2. Visualisations (`figures/`)
Target balance, numeric distributions, boxplots by outcome, categorical counts and disease rate per category,
risk by age/BMI/blood-pressure group, BP scatter + age density, pairplot, interaction heatmaps
(cholesterol x glucose, age x BP).

Key insights:
- **Classes are balanced** (34,642 healthy vs 33,924 diseased), so accuracy is a fair metric.
- **Age**: disease rate climbs from 22.5% (<=40) to 68.9% (60+).
- **Blood pressure** is the strongest signal: 22% disease in normal BP, 38% elevated, **78% hypertension**.
- **BMI**: 40% (normal) -> 51% (overweight) -> 68% (obese II+).
- **Cholesterol**: 43.5% / 59.6% / 76.2% for levels 1 / 2 / 3. Glucose shows a weaker version of the same trend.
- Inactive people have a somewhat higher rate (53.3% vs 48.5%).
- Smoking and alcohol show *no* positive association here (46.9% vs 49.7% smoker/non-smoker). This is likely due to
  self-reporting bias in the dataset, not evidence that smoking is harmless.
- Gender barely differs in disease rate (49.2% vs 50.0%).

### 3. Correlation matrix (`figures/10_correlation_matrix.png`)
Correlation with the target: `ap_hi` 0.43, `ap_lo` 0.34, `age` 0.24, `cholesterol` 0.22, `bmi` 0.19, `weight` 0.18,
`gluc` 0.09; `smoke`, `alco`, `height`, `gender` are ~0. Systolic/diastolic BP are strongly inter-correlated, as are
weight and BMI.

### 4. Model comparison (20% held-out test set, 13,714 patients)

| Model | CV acc. | Test accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.7350 | **0.7339** | 0.756 | 0.683 | 0.718 | **0.803** |
| SVM (RBF) | 0.7349 | 0.7332 | 0.756 | 0.680 | 0.716 | 0.791 |
| Decision Tree | 0.7306 | 0.7307 | 0.755 | 0.675 | 0.713 | 0.797 |
| Logistic Regression | 0.7273 | 0.7285 | 0.750 | 0.677 | 0.712 | 0.792 |
| KNN (k=25) | 0.7237 | 0.7222 | 0.735 | 0.686 | 0.710 | 0.785 |

Notes: SVM was trained on a 20,000-row subsample (and cross-validated on 10,000) because kernel SVM is slow on
55k rows. Differences between models are small (about 1 percentage point); the data seems to cap out around 73-74%.

### 5. Final model
**Random Forest** (300 trees, max depth 10, min leaf 5) scored best on accuracy and ROC-AUC. A randomised hyper-parameter
search did not beat it (tuned: 73.34% test accuracy), so the original was kept and saved to `models/best_model.joblib`.

Final test performance: **accuracy 73.4%, ROC-AUC 0.80**, precision 0.76 / recall 0.68 for the disease class.
Feature importance: systolic BP (45%), diastolic BP (21%), age (12%), cholesterol (9%), BMI (5%).

## Limitations
- ~73% accuracy is not clinical-grade. The data has no ECG, family history, or lab values, and BP/lifestyle fields are
  noisy and partly self-reported.
- Recall for disease is 68%, so roughly 1 in 3 cases is missed; a real screening tool would favour higher recall.
- This is an educational project, **not medical advice**.

## Ideas for improvement
Gradient boosting (XGBoost/LightGBM), threshold tuning for recall, SHAP explanations, full-data SVM grid search.
