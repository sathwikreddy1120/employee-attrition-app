import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Title
st.title("Employee Attrition Prediction App")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("hr_data.csv")
    return df

df = load_data()
st.subheader("HR Dataset Preview")
st.write(df.head())

# Basic input widgets
st.sidebar.header("Enter Employee Details")

satisfaction = st.sidebar.slider("Satisfaction Level", 0.0, 1.0, 0.5)
evaluation = st.sidebar.slider("Last Evaluation", 0.0, 1.0, 0.5)
projects = st.sidebar.slider("Number of Projects", 1, 10, 3)
average_monthly_hours = st.sidebar.slider("Average Monthly Hours", 90, 310, 160)
time_spent = st.sidebar.slider("Years at Company", 1, 10, 3)

# Predict button
if st.sidebar.button("Predict Attrition"):
    # Preprocessing
    X = df[["satisfaction_level", "last_evaluation", "number_project", 
            "average_montly_hours", "time_spend_company"]]
    y = df["left"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    input_data = np.array([[satisfaction, evaluation, projects, average_monthly_hours, time_spent]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The employee is likely to leave.")
    else:
        st.success("✅ The employee is likely to stay.")

