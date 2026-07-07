import streamlit as st
import pandas as pd
import joblib

st.title("💰 Loan Risk Calculator - AI Based System")

# Load model
model = joblib.load("Loan_Risk_Model.pkl")

uploaded_file = st.file_uploader(
    "Upload Excel or CSV File",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    try:
        uploaded_file.seek(0)

        # =========================
        # READ FILE
        # =========================
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        else:
            st.error("Unsupported file format!")
            st.stop()

        st.subheader("📄 Input Data Preview")
        st.dataframe(df.head())

        # =========================
        # AI PREDICTION
        # =========================
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df["AI Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]

        df["AI Probability"] = (probabilities * 100).round(1).astype(str) + "%"

        # =========================
        # RISK LEVEL
        # =========================
        def risk_level(p):
            if p < 0.3:
                return "Low"
            elif p < 0.7:
                return "Medium"
            else:
                return "High"

        df["Risk Level"] = pd.Series(probabilities).apply(risk_level)

        # =========================
        # BUSINESS DECISION
        # =========================
        def business_decision(risk):
            if risk == "Low":
                return "Approve"
            elif risk == "Medium":
                return "Review"
            else:
                return "Reject"

        df["Business Decision"] = df["Risk Level"].apply(business_decision)

        # =========================
        # REASON
        # =========================
        def reason(row):
            if row["Risk Level"] == "High":
                return "High risk due to weak credit behavior"
            elif row["Risk Level"] == "Medium":
                return "Moderate risk, needs manual review"
            else:
                return "Low risk, strong financial profile"

        df["Reason"] = df.apply(reason, axis=1)

        # =========================
        # FINAL COLUMN ORDER
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
            "Reason"
        ]

        df = df[final_columns]

        # =========================
        # REPORT
        # =========================
        st.subheader("📊 AI Loan Risk Business Report")
        st.dataframe(df, use_container_width=True)

        # =========================
        # SUMMARY REPORT
        # =========================
        st.subheader("📈 Summary Report")

        summary = df["Risk Level"].value_counts().reset_index()
        summary.columns = ["Risk Level", "Count"]

        st.dataframe(summary, use_container_width=True)

        # =========================
        # DOWNLOAD
        # =========================
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Full Report",
            csv,
            "loan_risk_report.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"File processing error: {e}")