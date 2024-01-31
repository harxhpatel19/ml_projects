import pickle
import streamlit as st
import os
import numpy as np

model_path = 'https://github.com/harxhpatel19/ml_projects/blob/main/heart_disease_data.csv'

if os.path.exists(model_path):
    heart_disease_model = pickle.load(open(model_path, 'rb'))
else:
    st.error("Error: Model file not found.")

# sidebar for navigation
selected = st.sidebar.selectbox('Heart Disease Prediction System', ['Heart Disease Prediction'])

if selected == 'Heart Disease Prediction':
    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age', type='number')

    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])

    # Convert sex to numeric representation
    sex_mapping = {'Male': 1, 'Female': 0}
    sex = sex_mapping[sex]

    with col3:
        cp = st.text_input('Chest Pain types', type='number')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure', type='number')

    # Repeat the same process for other input features...

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart_prediction = heart_disease_model.predict(input_data)

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

