import streamlit as st
import pickle
import numpy as np

# Load Models
housing_price_model = pickle.load(open('housing_price_model.pkl', 'rb'))
bank_churn_model = pickle.load(open('bank_churn_model.pkl', 'rb'))

# Housing Price Prediction
st.title('Housing Price Prediction')
st.write('Enter the following details to predict the house price:')

overall_qual = st.number_input('Overall Quality', min_value=1, max_value=10, value=5)
gr_liv_area = st.number_input('Ground Living Area (sqft)', value=1500)
garage_cars = st.number_input('Number of Garage Cars', min_value=0, max_value=4, value=1)
full_bath = st.number_input('Number of Full Bathrooms', min_value=0, max_value=4, value=2)

if st.button('Predict House Price'):
    features = np.array([[overall_qual, gr_liv_area, garage_cars, full_bath]])
    price_prediction = housing_price_model.predict(features)
    st.write(f'Predicted House Price: ${price_prediction[0]:,.2f}')

# Bank Churn Prediction
st.title('Bank Churn Prediction')
st.write('Enter the following details to predict if a customer will churn:')

credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance', value=50000.0)
estimated_salary = st.number_input('Estimated Salary', value=70000.0)

if st.button('Predict Churn'):
    features = np.array([[credit_score, age, balance, estimated_salary]])
    churn_prediction = bank_churn_model.predict(features)
    churn_prob = bank_churn_model.predict_proba(features)[0][1]
    st.write(f'Predicted Churn: {"Yes" if churn_prediction[0] else "No"}')
    st.write(f'Churn Probability: {churn_prob:.2f}')
