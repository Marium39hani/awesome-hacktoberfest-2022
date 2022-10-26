#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_boston
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[7]:


df=pd.read_csv('The_Grades_Dataset.csv')
df


# In[8]:


newdf=df.drop(columns=['HS-205/20','MT-222', 'EE-222', 'MT-224', 'CS-210', 'CS-211', 'CS-203', 'CS-214','EE-217', 'CS-212', 'CS-215', 'MT-331', 'EF-303', 'HS-304', 'CS-301','CS-302', 'TC-383', 'MT-442', 'EL-332', 'CS-318', 'CS-306', 'CS-312','CS-317', 'CS-403', 'CS-421', 'CS-406', 'CS-414', 'CS-419', 'CS-423','CS-412'])
newdf


# In[9]:


newdf['PH-121'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)
newdf['HS-101'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)
newdf['CY-105'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)
newdf['HS-105/12'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)
newdf['MT-111'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)
newdf['CS-105'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)
newdf['EL-102'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)
newdf['EE-119'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)
newdf['ME-107'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)
newdf['CS-107'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)
newdf['CS-106'].replace(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F','P','IP','X','I','W','WU'],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], inplace=True)


# In[10]:


newdf


# In[13]:


# interpolation
newdf.interpolate()
newdf.dropna(axis = 0,inplace=True)
newdf


# In[ ]:




