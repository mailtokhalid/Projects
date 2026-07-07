import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE TITLE
# =========================
st.set_page_config(page_title="Loan Risk Calculator", layout="wide")

st.title("💰 Loan Risk Calculator - AI Based System")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("Loan_Risk_Model.pkl")

# =========================
# BUSINESS LOGIC
# =========================

def risk_level(prob):
    if prob < 0.30:
        return "Low"
    elif prob < 0.70:
        return "Medium"
    else:
        return "High"


def make_risk_statement(row):

    reasons = []

    dti = row["LoanAmountRequested"] / (row["AnnualIncome"] + 1)

    # Credit Behaviour
    if row["CreditScore"] < 600:
        reasons.append("Low credit score")

    if row["LatePaymentsLastYear"] > 2:
        reasons.append("Frequent late payments")

    # Debt Burden
    if row["ExistingLoansCount"] >= 3:
        reasons.append("Multiple existing loans")

    if dti > 3:
        reasons.append("High debt-to-income ratio")

    # Employment
    if row["EmploymentStatus"] in ["Unemployed", "Student"]:
        reasons.append("Employment status increases repayment risk")

    if not reasons:
        return "Low risk based on current customer profile"

    return "; ".join(reasons)


def make_decision(row):

    dti = row["LoanAmountRequested"] / (row["AnnualIncome"] + 1)

    # Hard Reject
    if (
        row["CreditScore"] < 550
        or row["LatePaymentsLastYear"] >= 4
        or dti > 5
    ):
        return "Reject"

    # Manual Review
    if (
        row["CreditScore"] < 650
        or row["LatePaymentsLastYear"] >= 2
        or row["ExistingLoansCount"] >= 3
        or dti > 3
        or row["EmploymentStatus"] in ["Unemployed", "Student", "Retired"]
    ):
        return "Review Manually"

    return "Approve"


# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Excel or CSV File",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    try:

        uploaded_file.seek(0)

        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📄 Input Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # =========================
        # AI PREDICTION
        # =========================

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df["AI Prediction"] = ["Yes" if x == 1 else "No" for x in predictions]

        df["AI Probability"] = (
            probabilities * 100
        ).round(1).astype(str) + "%"

        df["Risk Level"] = pd.Series(probabilities).apply(risk_level)

        # =========================
        # BUSINESS RULES
        # =========================

        df["Business Decision"] = df.apply(make_decision, axis=1)

        df["Reason"] = df.apply(make_risk_statement, axis=1)

        # =========================
        # FINAL REPORT
        # =========================

        final_columns = [
            "CustomerID",
            "Name",
            "Age",
            "Gender",
            "MaritalStatus",
            "EducationLevel",
            "EmploymentStatus",
            "AnnualIncome",
            "LoanAmountRequested",
            "PurposeOfLoan",
            "CreditScore",
            "ExistingLoansCount",
            "LatePaymentsLastYear",
            "LoanApproved",
            "AI Prediction",
            "AI Probability",
            "Risk Level",
            "Business Decision",
            "Reason",
        ]

        report = df[final_columns]

        st.subheader("📊 AI Loan Risk Business Report")
        st.dataframe(report, use_container_width=True)

        # =========================
        # SUMMARY REPORT
        # =========================

        st.subheader("📈 Summary Report")

        summary = (
            report["Business Decision"]
            .value_counts()
            .reset_index()
        )

        summary.columns = [
            "Business Decision",
            "Count"
        ]

        st.dataframe(summary, use_container_width=True)

        # =========================
        # DOWNLOAD
        # =========================

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Report",
            csv,
            "Loan_Risk_Report.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")