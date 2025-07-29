import streamlit as st
import requests

#specify the FastAPI URL containing the predict path
#Note: this will access the predict api path created in app.py
API_URL = "http://localhost:8000/predict"

st.title("Insurance Premium Category Predictor")
st.markdown("Enter your details below:")

#Input fields
age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight(kg)", min_value = 1.0, value = 60.0)
height = st.number_input("Height(m)", min_value= 1.0,max_value=2.7, value=1.87)
income_lpa = st.number_input("Income (LPA)", min_value = 0.1, value = 10.0)
smoker = st.selectbox("Do you smoke?",options=[True,False])
city = st.text_input("City", value = "Mumbai")
occupation = st.selectbox("Occupation", options=['retired', 'freelancer', 'student', 'government_job', 'business_owner', 'unemployed', 'private_job'])

if st.button("Predict Premium Category"):
    #store input data in a json
    input_data = {
        "age":age,
        "weight":weight,
        "height":height,
        "income_lpa":income_lpa,
        "smoker":smoker,
        "city":city,
        "occupation":occupation
    }

    try:
        #send post request to /predict API in FastAPI server using input data as json
        response = requests.post(API_URL, json=input_data)
        
        #if we received correct response, display the same, as per our FastAPI predict function definition, we should get a status code of 200 and the predicted insurance category 
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Insurance Premium Category: **{result['predicted_category']}**")
        
        else:
            #print out the error status code (which will be different than 200) after removing the error text message
            st.error(f"API Error: {response.status_code} - {response.text}")
            
    
    except requests.exceptions.ConnectionError:
        #error indicates FastAPI server was not started prior to running the streamlit app
        st.error("Could not connect to FastAPI server, make sure FastAPI is already running on port 8000")
