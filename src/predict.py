"""Predict cardiovascular disease for one patient using the saved model.
Usage: python src/predict.py
"""
import os, joblib, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bundle = joblib.load(f"{ROOT}/models/best_model.joblib")

patient = dict(age=55, gender=1, height=170, weight=88, ap_hi=150, ap_lo=95,
               cholesterol=3, gluc=1, smoke=0, alco=0, active=1)   # gender: 0=female, 1=male
patient["bmi"] = patient["weight"] / (patient["height"] / 100) ** 2
X = pd.DataFrame([patient])[bundle["features"]]
model = bundle["model"]
pred = int(model.predict(X)[0])
print(f"Model: {bundle['name']}")
if hasattr(model, "predict_proba"):
    print(f"Estimated probability of cardiovascular disease: {model.predict_proba(X)[0, 1]:.1%}")
print("Prediction:", "HIGH RISK - disease likely" if pred else "LOW RISK - disease unlikely")
print("(Educational project, not medical advice.)")
