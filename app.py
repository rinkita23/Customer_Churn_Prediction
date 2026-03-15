import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model files
model = pickle.load(open("decision_tree_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
model_columns = pickle.load(open("model_columns.pkl","rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

st.write("Enter customer information to predict whether they are likely to churn.")

# Layout columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender",["Male","Female"])
    tenure = st.slider("Tenure (Months)",0,72,12)
    monthly_charges = st.slider("Monthly Charges",0,200,70)

with col2:
    contract = st.selectbox("Contract Type",["Month-to-month","One year","Two year"])
    internet = st.selectbox("Internet Service",["DSL","Fiber optic","None"])
    payment = st.selectbox("Payment Method",
                           ["Electronic check","Mailed check",
                            "Bank transfer","Credit card"])

if st.button("🔍 Predict Churn"):

    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges
    }

    input_df = pd.DataFrame([input_dict])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    scaled_data = scaler.transform(input_df)

    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠ Customer Likely to Churn\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer Likely to Stay\n\nProbability: {1-probability:.2f}")

    # Feature Importance
    st.subheader("Feature Importance")

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": model_columns,
        "Importance": importance
    }).sort_values("Importance", ascending=False).head(10)

    fig, ax = plt.subplots()

    ax.barh(feature_importance["Feature"], feature_importance["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)

st.markdown("---")
st.caption("Machine Learning Model: Decision Tree | Built with Streamlit")