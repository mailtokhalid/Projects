"""
Loan Risk Calculator - Hugging Face Spaces app (Gradio)
Upload a CSV of loan applicants, get back a scored CSV with predictions,
risk scores, and approval decisions.
"""

import joblib
import pandas as pd
import numpy as np
import gradio as gr

try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    def GPU_DECORATOR(fn):
        return fn

MODEL_PATH = "loan_risk_calculator.pkl"
pipeline = joblib.load(MODEL_PATH)

FEATURE_ORDER = [
    "loan_id", "age", "gender", "marital_status", "dependents", "education",
    "employment_status", "employment_years", "annual_income", "credit_score",
    "previous_default", "existing_loans_count", "loan_purpose", "loan_amount",
    "loan_term_months", "interest_rate", "property_ownership",
    "savings_balance", "collateral_value", "debt_to_income_ratio",
]

DECISION_THRESHOLD = 0.5


def calculate_loan_decision(row):
    """
    Simple rule-based risk score, separate from the ML model.
    Replace this with your own logic if you have a different version.
    """
    score = 0
    reasons = []

    if row["credit_score"] >= 750:
        score += 30
    elif row["credit_score"] >= 650:
        score += 20
    elif row["credit_score"] >= 550:
        score += 10
    else:
        reasons.append("Low credit score")

    if row["debt_to_income_ratio"] <= 0.3:
        score += 25
    elif row["debt_to_income_ratio"] <= 0.5:
        score += 15
    else:
        reasons.append("High debt-to-income ratio")

    if row["previous_default"] == 0:
        score += 20
    else:
        reasons.append("Previous default on record")

    if row["employment_years"] >= 3:
        score += 15
    else:
        reasons.append("Limited employment history")

    if row["existing_loans_count"] <= 2:
        score += 10
    else:
        reasons.append("Many existing loans")

    if score >= 70:
        risk_level = "Low Risk"
        approval_status = "Approved"
    elif score >= 45:
        risk_level = "Medium Risk"
        approval_status = "Review Required"
    else:
        risk_level = "High Risk"
        approval_status = "Rejected"

    reason_text = "; ".join(reasons) if reasons else "Meets all standard criteria"

    return pd.Series({
        "RiskScore": score,
        "RiskLevel": risk_level,
        "ApprovalStatus": approval_status,
        "Reason": reason_text,
    })


@GPU_DECORATOR
def process_csv(file):
    if file is None:
        return None, "Please upload a CSV file."

    try:
        file_path = file.name if hasattr(file, "name") else file
        df = pd.read_csv(file_path)
    except Exception as e:
        return None, f"Could not read the CSV file: {e}"

    missing_cols = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing_cols:
        return None, f"CSV is missing required columns: {missing_cols}"

    X = df[FEATURE_ORDER].copy()

    try:
        probs = pipeline.predict_proba(X)[:, 1]
    except Exception as e:
        return None, f"Error running the model: {e}"

    df["AI_Prediction"] = pd.Series(
        (probs >= DECISION_THRESHOLD).astype(int), index=df.index
    ).map({0: "Rejected", 1: "Approved"})
    df["AI_Probability"] = (probs * 100).round(2)

    decision_cols = df.apply(calculate_loan_decision, axis=1)
    df[["RiskScore", "RiskLevel", "ApprovalStatus", "Reason"]] = decision_cols

    new_cols = ["AI_Prediction", "AI_Probability", "RiskScore", "RiskLevel", "ApprovalStatus", "Reason"]
    cols = [c for c in df.columns if c not in new_cols]
    if "loanapproval" in cols:
        idx = cols.index("loanapproval") + 1
    else:
        idx = len(cols)
    final_cols = cols[:idx] + new_cols + cols[idx:]
    df = df[final_cols]

    output_path = "scored_results.csv"
    df.to_csv(output_path, index=False)

    status = f"Done. Processed {len(df)} rows."
    return output_path, status


with gr.Blocks(title="Loan Risk Calculator") as demo:
    gr.Markdown("# Loan Risk Calculator")
    gr.Markdown(
        "Upload a CSV file of loan applicants. Each row will be scored by "
        "the AI model and given a risk level and approval status. "
        "Download the results as a new CSV file."
    )

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])

    process_btn = gr.Button("Process File", variant="primary")

    status_output = gr.Textbox(label="Status")
    file_output = gr.File(label="Download Results")

    process_btn.click(
        fn=process_csv,
        inputs=[file_input],
        outputs=[file_output, status_output],
    )

if __name__ == "__main__":
    demo.launch()