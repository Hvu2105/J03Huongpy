import streamlit as st
import pickle
import numpy as np

# Load Models
bank_churn_model = pickle.load(open('bank_churn_model.pkl', 'rb'))

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

