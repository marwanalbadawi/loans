import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.image("nnn.jpg", width=200)
st.markdown("**Student**: Marwan Mustafa Badawi Suluiman. **Class**: DS 401")
st.header("Prediction of Loan Acceptance")
st.subheader("This model will predict the acceptance or decline of a loan request, based on ML approach" )
st.text("Please fill the following three enteries only")

#st.button("Predict")
# Load the pre-trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Input collection
person_home_ownership = st.radio("Person_Home_Ownership", ["Own", "Rent", "Mortgage", "Others"])
loan_percent_income = st.slider("Loan_Percent_Income (Percentage of the loan from applicant's income)", 0.01, 30.0)
loan_int_rate = st.slider("Loan_Int_Rate (Interest rate of the loan)", 0.01, 30.0)

# Preprocessing
ownership_mapping = {"Own": 0, "Rent": 1, "Mortgage": 2, "Others": 3}
ownership_encoded = ownership_mapping[person_home_ownership]
input_data = np.array([[ownership_encoded, loan_percent_income, loan_int_rate]])

# Prediction and Custom Text
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("Loan Status Prediction: Approved ✅")
        st.write("The accuracy of true acceptance is almost 70%. Model is to be apdated soon with better performance.")
    else:
        st.write("Loan Status Prediction: Rejected ❌")
        st.write(" This is 97% accurate.")
st.subheader("Thank you!")
st.text("Check out Machinfy on social media for more usefull AI courses & applications")
