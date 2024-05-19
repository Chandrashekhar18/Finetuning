#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[2]:


dataset=pd.read_csv(r"D:\ml project 1\car_prices.csv")


# In[3]:


dataset.head(10)


# In[4]:


from sklearn.preprocessing import LabelEncoder
column_to_encode = 'fuel'
label_encoder = LabelEncoder()
dataset[column_to_encode] = label_encoder.fit_transform(dataset[column_to_encode])
dataset


# In[5]:


dataset


# In[6]:


columns_to_check = [ 'mileage','vol_engine','price','fuel_encoded','model_encoded','city_encoded']
def remove_outliers_iqr(dataset, columns):
    for column in columns:
        Q1 = dataset[column].quantile(0.25)
        Q3 = dataset[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset = dataset[(dataset[column] >= lower_bound) & (dataset[column] <= upper_bound)]
    return dataset
dataset_cleaned = remove_outliers_iqr(dataset, columns_to_check)


# In[ ]:


dataset_cleaned

