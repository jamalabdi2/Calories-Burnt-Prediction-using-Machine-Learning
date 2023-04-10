from fastapi import FastAPI
from pydantic import BaseModel
import xgboost 
import pandas as pd
import uvicorn
import os
import pickle
import numpy as np

app = FastAPI()
model_path = '/Users/jamal/Desktop/SAP/Project3/FastApi/Saved Models/xgbregressor_model.pickle'
# model_path  = os.environ.get('MODEL_PATH')
# if not model_path:
#    raise ValueError('Model_path environment variable not set')
# else:
#    model_path = os.environ["MODEL_PATH"] = "/Users/jamal/Desktop/SAP/Project3/FastApi/Saved Models/xgbregressor_model.pickle"

try:
  with open(model_path,'rb') as f:
    model = pickle.load(f)
    print('Model have been successfully loaded')
except Exception as e:
  print('Problem loading the pickle file:{e} ')

class Features(BaseModel):
    Gender: str
    Age: int
    Height: float
    Weight: float
    Duration: float
    Heart_Rate: float
    Body_Temp: float

def encode_gender(df):
   df['Gender'] = df['Gender'].map({'female':0,'male':1})
   return df

def calculate_bmi(df):
   df['BMI'] = df['Weight']/((df['Height']/100)**2)
   return df

def categorize_bmi(df):
   bins=[0,18.5,25,30,40]
   labels=['Underweight','Normal Weight','Overweight','Obese']
   df['BMI Category'] = pd.cut(df['BMI'],bins=bins,labels=labels)
   return df

def categorize_age(df):
   bins = [0,14,24,64,100]
   labels = ['Children','Youth','Adult','Old']
   df['Age Group'] = pd.cut(df['Age'],bins=bins,labels=labels)
   return df

def one_hot_encode(df):
    df = pd.get_dummies(df, columns=['Age Group','BMI Category'])
    return df
   

def preprocess_data(df:pd.DataFrame):

  df = encode_gender(df)
  df = calculate_bmi(df)
  df = categorize_bmi(df)
  df = categorize_age(df)
  df = one_hot_encode(df)
  df = df.astype(np.float32)
  return df 

                             
@app.post('/predict')
def predict(features: Features):
    data = preprocess_data(pd.DataFrame([features.dict()]))
    prediction = model.predict(data).tolist()
    return {'prediction': prediction}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
