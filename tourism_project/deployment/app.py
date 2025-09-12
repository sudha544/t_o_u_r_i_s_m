import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import HfApi, create_repo
from google.colab import userdata # This might not work directly in a deployed app, need to handle credentials securely

st.title("Wellness Tourism Package Purchase Prediction")

st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Input fields for customer details
customer_id = st.number_input("CustomerID", min_value=0)
age = st.number_input("Age", min_value=0, max_value=120)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business"]) # Add other relevant occupations
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number of People Visiting", min_value=1)
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number of Trips Annually", min_value=0)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0)
designation = st.text_input("Designation") # Consider making this a selectbox if possible
monthly_income = st.number_input("Monthly Income", min_value=0.0)

# Input fields for customer interaction data
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe"]) # Add other relevant products
number_of_followups = st.number_input("Number of Follow-ups", min_value=0)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0)

# Create a dictionary from inputs (excluding ProdTaken as it's the target)
input_data = {
    'CustomerID': customer_id,
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': number_of_followups,
    'DurationOfPitch': duration_of_pitch,
}

# Convert input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Save Input Data"):
    # Define the path to save the input data
    input_data_path = "collected_input_data.csv"
    input_df.to_csv(input_data_path, index=False)

    st.success(f"Input data saved locally to {input_data_path}")

    # --- Upload the saved data to Hugging Face Space (Requires HF credentials) ---
    try:
        # Note: Using userdata.get('HF_TOKEN') directly in a deployed Streamlit app
        # hosted outside Colab is not recommended for security.
        # Consider more secure methods for handling credentials in production.
        HF_TOKEN = userdata.get('HF_TOKEN') # Get Hugging Face token from Colab secrets
        HF_USERNAME = "sudha1726" # Replace with your Hugging Face username
        HF_SPACE_REPO = "tourism-package-purchase-inputs" # Replace with your desired Hugging Face Space repo name

        api = HfApi()

        # Create the repository on Hugging Face Hub if it doesn't exist
        create_repo(repo_id=f"{HF_USERNAME}/{HF_SPACE_REPO}", repo_type="space", exist_ok=True, token=HF_TOKEN)


        # Upload the collected data file to the Hugging Face Space repository
        api.upload_file(
            path_or_fileobj=input_data_path,
            path_in_repo=input_data_path,
            repo_id=f"{HF_USERNAME}/{HF_SPACE_REPO}",
            repo_type="space",
            token=HF_TOKEN,
        )

        st.success(f"Input data uploaded to Hugging Face Space: https://huggingface.co/spaces/{HF_USERNAME}/{HF_SPACE_REPO}/blob/main/{input_data_path}")

    except Exception as e:
        st.error(f"Error uploading data to Hugging Face: {e}")
