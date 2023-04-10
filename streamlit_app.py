import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import xgboost
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
warnings.filterwarnings('ignore')
import os


# streamlit ui 
# title
st.title('Calories Burnt Prediction :mechanical_arm:')
st.write("This Web App, aims to predict the amount of calories burnt during physical activity based on various factors such as the your gender, height, weight, and the type of activity performed. The project uses machine learning algorithms to achieve this goal.")
st.sidebar.header("Please Provide The Following Data")

#Gender	Age	Height	Weight	Duration	Heart_Rate	Body_Temp	Calories
def user_input():
    """
    Collect user inputs through Streamlit sliders and radio button and return the information as a Pandas dataframe.

    Returns:
    Pandas DataFrame: A dataframe containing the collected information of age, BMI, heart rate, duration, body temperature, and gender. Gender is transformed to a numerical value (0 for male and 1 for female/non-binary) for easier processing.
    """

    global age,heart_rate,duration,body_temperature,gender,name
    name = st.sidebar.text_input('Name:')
    age = st.sidebar.slider('Age:',5,100,20)
    height = st.sidebar.slider('Height:',5,200,20)
    Weight = st.sidebar.slider('Weight:',5,200,20)
    heart_rate = st.sidebar.slider('Heart Rate',60,130,70)
    duration = st.sidebar.slider('Duration in Minute',0,60,20)
    body_temperature = st.sidebar.slider('Body Temperature',36,40,37)
    gender = st.sidebar.radio('Gender: ',('Male','Female'))
    
   #put information collected from user to a dictionary(for display purpose only)
    user_data_dictionary = {
        'Name':name,
        'Gender':gender,
        'Age':age,
        'Height':height,
        'Weight':Weight,
        'Heart Rate':heart_rate,
        'Body Temperature':body_temperature,
        'Duration':duration,
    }

    #{'Gender':{'male':0,'female':1}}
    # label encoding for gender 
  
    if gender =='Male':
        gender = 1
    else:
        gender=0

    #Gender	Age	Height	Weight	Duration	Heart_Rate	Body_Temp
    model_features_dictionary= {
        'Gender':gender,
        'Age':age,
        'Height':height,
        'Weight':Weight,
        'Duration':duration,
        'Heart_Rate':heart_rate,
        'Body_Temp':body_temperature,
    }

    #turn the dictionaries to a pandas dataframe
    information_collected = pd.DataFrame([user_data_dictionary])
    model_input_features = pd.DataFrame([model_features_dictionary])
    return information_collected,model_input_features

user_display,model_features = user_input()
st.header("Your information")
st.dataframe(user_display)

#preprocessing
def preprocess_data(df):
  '''
  preprocess data for fastapi
  
  '''
  #df['Gender'] = df['Gender'].map({'Female':0,'Male':1})
  df['BMI'] = df['Weight']/((df['Height']/100)**2)
  df['BMI Category'] = pd.cut(df['BMI'],bins = [0,18.5,25,30,100],labels = ['Underweight','Normal Weight','Overweight','Obese'])
  df['Age Group'] = pd.cut(df['Age'],bins= [0,14,24,64,100], labels = ['Children','Youth','Adult','Old'])
  df = pd.get_dummies(df,columns = ['Age Group','BMI Category'])
  return df
processed_data = preprocess_data(model_features)
st.header("Processed Data")
st.dataframe(processed_data)

def load_model(model_path:str):
    if not isinstance(model_path,str):
        raise ValueError(f'The model path: {model_path} must be a string')
    
    if not os.path.isfile(model_path):
        raise ValueError(f'The file {model_path} does not exits')
    
    with open(model_path,'rb') as file:
        model = pickle.load(file)
        if isinstance(model,xgboost.XGBRegressor):
            print('The model is XGBRegressor')
        else:
            print('The model is not XGBRegressor')

    return model
model_path = '/Users/jamal/Golang/saved_model/xgbregressor_model.pickle'
model = load_model(model_path)
st.header("Prediction : ")



prediction = model.predict(processed_data)
prediction_str = str(prediction)
st.subheader("Estimated Calories Burnt:" + prediction_str + "calories")

print('model prediction: ',prediction)
