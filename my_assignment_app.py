import streamlit as st
import pandas as pd
from joblib import load
import dill

with open("C:/Users/Lenovo/Downloads/pipeline_Assignment.pkl", 'rb') as file:
    model = dill.load(file)
my_feature_dict = load("C:/Users/Lenovo/Downloads/my_feature_dict_Assignment.pkl")

st.header("MLOPS assignment")
st.subheader("Created by: Saleem Hassan Vazir")


st.subheader('Categorical Features') 
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals={}
for i, col in enumerate(categorical_input.get('Column Name').values()): 
    categorical_input_vals[col] = st.selectbox(col, categorical_input.get('Members')[i])

st.subheader('Numerical Features') 
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals={}
for col in numerical_input.get('Column Name'): 
    numerical_input_vals[col] = st.number_input(col)
    

def predict_churn(data):
    prediction = model.predict(data)
    return prediction

input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))
input_data = pd.DataFrame.from_dict(input_data,orient='index').T

if st.button("Predict"):
    prediction = predict_churn(input_data)[0]
    translation_dict = {"Yes": "Expected", "No": "Not Expected"}
    prediction_translate = translation_dict.get(prediction)
    st.write(f"The Prediction is '{prediction}'.This means that the customer is '{prediction_translate}' to churn.")
