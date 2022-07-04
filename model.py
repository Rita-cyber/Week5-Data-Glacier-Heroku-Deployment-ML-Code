#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/datagy/data/main/insurance.csv')
print(df.head())


# In[2]:


# Exploring the dataset
print(df.info())


# In[3]:


df['region'].unique()


# In[4]:


df['sex'].unique()


# In[16]:


# Creating new variables
df['smoker_int'] = df['smoker'].map({'yes':1, 'no':0})
df['sex_int'] = df['sex'].map({'female':1, 'male':0})
df['region_int'] = df['region'].map({'southwest':1, 'southeast':2,'northwest':3,'northeast':4})

X = df[['age', 'sex_int', 'bmi','children','smoker_int']]
y = df['charges']

import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()

#Fitting model with trainig data
lm.fit(X_train, y_train)

# Saving model to disk
pickle.dump(lm, open('model.pkl','wb'))


# In[ ]:




