import streamlit as st
import sys
import os

# Make sure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_churn

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict the likelihood they will churn.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_bak = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple,
        'InternetService': internet,
        'OnlineSecurity': online_sec,
        'OnlineBackup': online_bak,
        'DeviceProtection': device,
        'TechSupport': tech,
        'StreamingTV': tv,
        'StreamingMovies': movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
    }

    prob = predict_churn(input_dict)
    pct = prob * 100

    st.divider()
    if prob >= 0.7:
        st.error(f"High churn risk: {pct:.1f}%")
        st.write("This customer is very likely to leave. Priority retention offer recommended.")
    elif prob >= 0.4:
        st.warning(f"Medium churn risk: {pct:.1f}%")
        st.write("This customer shows signs of risk. Consider a proactive check-in.")
    else:
        st.success(f"Low churn risk: {pct:.1f}%")
        st.write("This customer is likely to stay.")

    st.progress(float(prob))