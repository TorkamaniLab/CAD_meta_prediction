#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv("cardio_train.csv", sep=";")
categorical_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
df[categorical_cols] = df[categorical_cols].astype("object")
df.to_pickle("typed_cardio_train.pkl")
