"""
Loan Risk Calculator - Hugging Face Spaces app (Gradio)
Loads a pre-trained sklearn Pipeline (preprocessor + XGBClassifier) and
serves loan approval predictions through a simple web form.
"""

import joblib
import pandas as pd
import gradio as gr

# Only needed if this Space's hardware is set to ZeroGPU. On CPU-only
# hardware this import will fail, which is fine — we fall back to a
# no-op decorator so the app runs identically either way.
try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    def GPU_DECORATOR(fn):
        return fn

MODEL_PATH = "loan_risk_calculator.pkl"

# -----------------------------------------------------------------
# Load pipeline once at startup
# -----------------------------------------------------------------
pipeline = joblib.load(MODEL_PATH)

# Exact column order the pipeline was fitted on (from feature_names_in_)
FEATURE_ORDER = [
    "loan_id", "age", "gender", "marital_status", "dependents", "education",
    "employment_status", "employment_years", "annual_income", "credit_score",
    "previous_default", "existing_loans_count", "loan_purpose", "loan_amount",
    "loan_term_months", "interest_rate", "property_ownership",
    "savings_balance", "collateral_value", "debt_to_income_ratio",
]

# Decision threshold — adjust here if you tuned a custom cutoff
# (see earlier discussion on threshold tuning for imbalanced approve/reject costs)
DECISION_THRESHOLD = 0.5


@GPU_DECORATOR
def predict_loan_risk(
    loan_id, age, gender, marital_status, dependents, education,
    employment_status, employment_years, annual_income, credit_score,
    previous_default, existing_loans_count, loan_purpose, loan_amount,
    loan_term_months, interest_rate, property_ownership,
    savings_balance, collateral_value, debt_to_income_ratio,
):
    row = {
        "loan_id": loan_id or "LN999999",
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "dependents": dependents,
        "education": education,
        "employment_status": employment_status,
        "employment_years": employment_years,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "previous_default": 1 if previous_default == "Yes" else 0,
        "existing_loans_count": existing_loans_count,
        "loan_purpose": loan_purpose,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "property_ownership": property_ownership,
        "savings_balance": savings_balance,
        "collateral_value": collateral_value,
        "debt_to_income_ratio": debt_to_income_ratio,
    }

    X = pd.DataFrame([row], columns=FEATURE_ORDER)

    try:
        prob_approved = float(pipeline.predict_proba(X)[:, 1][0])
    except Exception as e:
        return f"Error running prediction: {e}", "", ""

    decision = "Approved" if prob_approved >= DECISION_THRESHOLD else "Rejected"

    if prob_approved >= 0.75 or prob_approved <= 0.25:
        risk_level = "Low Risk" if decision == "Approved" else "High Risk"
    else:
        risk_level = "Medium Risk"

    result_md = f"""
### Decision: **{decision}**
- Approval probability: **{prob_approved * 100:.2f}%**
- Risk level: **{risk_level}**
- Decision threshold used: {DECISION_THRESHOLD}
"""
    return decision, f"{prob_approved * 100:.2f}%", risk_level


with gr.Blocks(title="Loan Risk Calculator") as demo:
    gr.Markdown("# 🏦 Loan Risk Calculator")
    gr.Markdown(
        "Enter applicant details below to get an AI-powered loan approval "
        "prediction. This tool is for demonstration purposes and should not "
        "be used as the sole basis for real lending decisions."
    )

    with gr.Row():
        with gr.Column():
            loan_id = gr.Textbox(label="Loan ID (optional)", value="LN999999")
            age = gr.Number(label="Age", value=35, minimum=18, maximum=100)
            gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
            marital_status = gr.Dropdown(
                ["Single", "Married", "Divorced", "Widowed"],
                label="Marital Status", value="Single"
            )
            dependents = gr.Number(label="Dependents", value=0, minimum=0, maximum=10)
            education = gr.Dropdown(
                ["High School", "Associate", "Bachelor's", "Master's", "PhD"],
                label="Education", value="Bachelor's"
            )
            employment_status = gr.Dropdown(
                ["Employed", "Self-Employed", "Unemployed", "Retired"],
                label="Employment Status", value="Employed"
            )
            employment_years = gr.Number(label="Employment Years", value=5, minimum=0, maximum=50)
            annual_income = gr.Number(label="Annual Income ($)", value=60000, minimum=0)
            credit_score = gr.Slider(300, 850, value=680, label="Credit Score")

        with gr.Column():
            previous_default = gr.Radio(
                ["No", "Yes"], label="Previous Default on Record?", value="No"
            )
            existing_loans_count = gr.Number(label="Existing Loans Count", value=1, minimum=0, maximum=10)
            loan_purpose = gr.Dropdown(
                ["Home", "Auto", "Education", "Personal", "Business", "Medical", "Debt Consolidation"],
                label="Loan Purpose", value="Personal"
            )
            loan_amount = gr.Number(label="Loan Amount ($)", value=15000, minimum=0)
            loan_term_months = gr.Dropdown(
                [12, 24, 36, 48, 60, 84, 120, 180, 240, 360],
                label="Loan Term (months)", value=36
            )
            interest_rate = gr.Number(label="Interest Rate (%)", value=10.5, minimum=0, maximum=40)
            property_ownership = gr.Dropdown(
                ["Own", "Mortgage", "Rent"], label="Property Ownership", value="Rent"
            )
            savings_balance = gr.Number(label="Savings Balance ($)", value=5000, minimum=0)
            collateral_value = gr.Number(label="Collateral Value ($)", value=0, minimum=0)
            debt_to_income_ratio = gr.Number(
                label="Debt-to-Income Ratio (0-1.5)", value=0.3, minimum=0, maximum=1.5
            )

    submit_btn = gr.Button("Calculate Loan Risk", variant="primary")

    with gr.Row():
        decision_out = gr.Textbox(label="Decision")
        probability_out = gr.Textbox(label="Approval Probability")
        risk_out = gr.Textbox(label="Risk Level")

    submit_btn.click(
        fn=predict_loan_risk,
        inputs=[
            loan_id, age, gender, marital_status, dependents, education,
            employment_status, employment_years, annual_income, credit_score,
            previous_default, existing_loans_count, loan_purpose, loan_amount,
            loan_term_months, interest_rate, property_ownership,
            savings_balance, collateral_value, debt_to_income_ratio,
        ],
        outputs=[decision_out, probability_out, risk_out],
    )

if __name__ == "__main__":
    demo.launch()
