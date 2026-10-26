import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction Dashboard")
st.caption("Machine Learning-Based Employee Attrition Risk Scoring System")

# Load the saved employee risk data
DATA_FILE = "employee_attrition_risk_scores.csv"
MODEL_FILE = "attrition_model.pkl"

if not os.path.exists(DATA_FILE):
    st.error("Risk data file not found. Please upload the CSV file.")
    st.stop()

df = pd.read_csv(DATA_FILE)

# Summary metrics
total = len(df)
high = (df["Risk_Level"] == "High").sum()
medium = (df["Risk_Level"] == "Medium").sum()
low = (df["Risk_Level"] == "Low").sum()

attrition_count = (
    df["Attrition"].astype(str).str.strip().str.lower().isin(
        ["yes", "1", "true"]
    ).sum()
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Employees", total)
c2.metric("Recorded Attrition", attrition_count)
c3.metric("High Risk", high)
c4.metric("Medium Risk", medium)

st.divider()

# Risk distribution chart
st.subheader("Employee Risk Distribution")

risk_counts = (
    df["Risk_Level"].value_counts()
    .reindex(["Low", "Medium", "High"], fill_value=0)
)

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(risk_counts)
    st.caption("Number of employees in each risk category.")

with col2:
    st.subheader("Risk Category Summary")
    st.dataframe(
        risk_counts.rename_axis("Risk Level")
        .reset_index(name="Employees"),
        hide_index=True,
        use_container_width=True
    )

# Department analysis
if "Department" in df.columns:
    st.divider()
    st.subheader("Risk by Department")

    dept_risk = pd.crosstab(
        df["Department"],
        df["Risk_Level"]
    ).reindex(columns=["Low", "Medium", "High"], fill_value=0)

    st.bar_chart(dept_risk)

# Employee risk table
st.divider()
st.subheader("Employee Risk Details")

risk_filter = st.selectbox(
    "Filter by risk level",
    ["All", "Low", "Medium", "High"]
)

filtered_df = df.copy()

if risk_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Risk_Level"] == risk_filter
    ]

search_columns = [
    col for col in
    ["Age", "Department", "JobRole", "Attrition",
     "Attrition_Risk_Score", "Risk_Level"]
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[search_columns],
    use_container_width=True,
    hide_index=True
)

# Download filtered data
st.download_button(
    label="Download Employee Risk Data (CSV)",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="employee_attrition_risk_scores.csv",
    mime="text/csv"
)

st.divider()
st.caption(
    "Note: Risk scores are model estimates, not certainty about "
    "an employee's future behavior. Use them responsibly and "
    "avoid making employment decisions based solely on predictions."
)
