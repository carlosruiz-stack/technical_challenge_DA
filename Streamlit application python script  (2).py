#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


# In[2]:


dataset_1A = pd.read_csv('dataset_1A.csv')
dataset_1A


# In[3]:


dataset_1A['total_consumption'] = (dataset_1A['distance'] * dataset_1A['consume']) / 100
dataset_1A.head()


# In[5]:


X = dataset_1A.drop(columns=['total_consumption'])
y = dataset_1A['total_consumption']

data = pd.get_dummies(dataset_1A, columns=['gas_type'], drop_first=True)

encoder = OneHotEncoder(drop='first', sparse=False)
gas_type_encoded = encoder.fit_transform(data[['gas_type']])
gas_type_encoded_df = pd.DataFrame(gas_type_encoded, columns=encoder.get_feature_names(['gas_type']))

X_encoded = pd.concat([X, gas_type_encoded_df], axis=1)

model = LinearRegression()
model.fit(X_encoded, y)

st.title("Total Consumption Prediction")

st.sidebar.header("Predictor Variables")
distance = st.sidebar.number_input("Distance", value=1.0)
consume = st.sidebar.number_input("Consume", value=7.0)
speed = st.sidebar.number_input("Speed", value=40)
temp_inside = st.sidebar.number_input("Temperature Inside", value=20.0)
temp_outside = st.sidebar.number_input("Temperature Outside", value=10.0)
ac = st.sidebar.checkbox("AC")
rain = st.sidebar.checkbox("Rain")
sun = st.sidebar.checkbox("Sun")
refill_liters = st.sidebar.number_input("Refill Liters", value=0.0)
refill_gas = st.sidebar.number_input("Refill Gas", value=0.0)
gas_type = st.sidebar.checkbox("Gas Type")

ac = int(ac)
rain = int(rain)
sun = int(sun)
gas_type = int(gas_type)

input_data = [
    [
        distance, consume, speed, temp_inside, temp_outside,
        ac, rain, sun, refill_liters, refill_gas, gas_type
    ] + gas_type_encoded_df.iloc[0].tolist()
]

predicted_consumption = model.predict(input_data)

st.write("Predicted Total Consumption:", predicted_consumption[0])


# In[ ]:




