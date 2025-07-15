import tensorflow as tf 
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st 

model=load_model('model.h5')
with open('labelencoder.pkl','rb') as f:
    label=pickle.load(f)
with open('onehotencoder.pkl','rb') as f:
    onehot=pickle.load(f)
with open('stander.pkl','rb') as f:
    standscaler=pickle.load(f)

st.title('Customer Chun prediction')
CreditScore=st.number_input('CreditScore')
Geography=st.selectbox('Geography',onehot.categories_[0])
Gender=st.selectbox('Gender',label.classes_)
Age=st.slider('Age',18,92)
Tenure=st.slider('Tenure',1,10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider('NumOfProducts',1,4)
HasCrCard=st.selectbox('HasCrCard',[0,1])
IsActiveMember=st.selectbox('IsActiveMember',[0,1])
EstimatedSalary=st.number_input('EstimatedSalary')

input_data =pd.DataFrame( {
    'CreditScore': [CreditScore],
    'Gender': [label.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance':[Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
}
)
onehotencoder_df=onehot.transform([[Geography]]).toarray()
onehot_dataframe=pd.DataFrame(onehotencoder_df,columns=onehot.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.reset_index(drop=True),onehot_dataframe],axis=1)
input_scaled=standscaler.transform(input_data)
prediction=model.predict(input_data)
if prediction>0.5:
    st.write('The customer is likly churn')
else:
    st.write('The customer is not likly churn')