import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
data = pd.read_csv("Loan approval prediction.csv")
# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "View Data", "Visualize Data"])

if page == "Home":
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

elif page == "View Data":
    st.title("Raw Data Viewer")
    search_term = st.text_input("Search Data (Enter keywords or ID)")
    if search_term:
        filtered_data = data[data.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
        st.write(filtered_data)
    else:
        st.write(data)

elif page == "Visualize Data":
    st.title("Data Visualization")

    # Exclude the ID column
    selectable_columns = [col for col in data.columns if col.lower() != 'id']
    column_to_visualize = st.selectbox("Select Column to Visualize", selectable_columns)
    
    if column_to_visualize:
        # Adjust bin size for x-axis
        bin_size = st.slider("Select Bin Size (Interval for X-axis)", min_value=1, max_value=50, value=10)
        
        # Create a histogram and customize bar colors
        fig = px.histogram(
            data,
            x=column_to_visualize,
            title=f"Distribution of {column_to_visualize}",
            nbins=bin_size
        )
        
        # Update bar colors manually
        fig.update_traces(marker=dict(color=px.colors.qualitative.Set3 * 10))  # Repeated color palette for bars
        
        # Update layout for better visuals
        fig.update_layout(
            xaxis_title=f"{column_to_visualize}",
            yaxis_title="Count",
            plot_bgcolor='rgba(240, 240, 240, 0.9)',  # Light background
            title_font_size=18
        )
        
        st.plotly_chart(fig)
