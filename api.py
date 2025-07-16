import tensorflow as tf 
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
model=load_model('model.h5',compile=False)
with open('labelencoder.pkl','rb') as f:
    label=pickle.load(f)
with open('onehotencoder.pkl','rb') as f:
    onehot=pickle.load(f)
with open('stander.pkl','rb') as f:
    standscaler=pickle.load(f)

app=FastAPI()

class Model_select(BaseModel):
    CreditScore:int
    Geography:str
    Gender:str
    Age:int
    Tenure:int
    Balance:float
    NumOfProducts:int
    HasCrCard:int
    IsActiveMember:int
    EstimatedSalary:float

@app.get('/')
def strat_method():
   return {'meassage':'welcome to CustomerExist'}


@app.post('/predict')
def predict_model(data:Model_select):
   input_data=data.dict()
   Gender_label=label.transform([input_data['Gender']])[0]
   one=onehot.transform([[input_data['Geography']]]).toarray()
   names=[
      input_data['CreditScore'],
      Gender_label,
      input_data['Age'],
      input_data['Tenure'],
      input_data['Balance'],
      input_data['NumOfProducts'],
      input_data['HasCrCard'],
      input_data['IsActiveMember'],
      input_data['EstimatedSalary']
   
    ]
   names=pd.array(names).reshape(1,-1)
   final=np.concatenate((names,one),axis=1)
   sta=standscaler.transform(final)
   model_pre=model.predict(sta)
   #return {'customer is :',int(model_pre)}
   #print(int(model_pre))
   if model_pre==0:
        return {'customer is : Not Exist'}
   else:
       return{'customer is : Exist'}
 
 