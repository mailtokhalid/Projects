---
title: Loan Risk Calculator
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Loan Risk Calculator

An AI-powered loan approval predictor built on a scikit-learn preprocessing
pipeline + XGBoost classifier, served through a Gradio interface.

## Files

- `app.py` — Gradio web app
- `loan_risk_calculator.pkl` — trained sklearn Pipeline (preprocessor + XGBClassifier)
- `requirements.txt` — pinned dependencies matching the training environment

## Important notes

- This model was trained with **scikit-learn 1.6.1**. The `requirements.txt`
  pins this exact version — do not upgrade it, or the pickle will fail to
  load (newer scikit-learn versions changed internal `ColumnTransformer`
  class names).
- If the Space fails to start with an `xgboost` version-related error, check
  the exact xgboost version used at training time (`xgboost.__version__` in
  your training notebook/environment) and update the version pin in
  `requirements.txt` to match exactly.
- This tool is for demonstration/educational purposes only and should not be
  used as the sole basis for real lending decisions.

## Deploying

1. Create a new Space on Hugging Face (SDK: Gradio).
2. Upload all three files (`app.py`, `requirements.txt`, `loan_risk_calculator.pkl`) to the Space's repo root.
3. The Space will build automatically and launch `app.py`.
