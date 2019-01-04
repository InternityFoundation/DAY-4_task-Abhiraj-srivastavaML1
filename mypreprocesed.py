# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 12:34:11 2019

@author: SP Srivastava
"""
#importing the data
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
data=pd.read_csv('Data.csv')
X=data.iloc[:,:-1].values
Y=data.iloc[:,3].values
#handling hte n=missing values
from sklearn.preprocessing import Imputer 
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
#enconding the data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,0]=labelencoder_x.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
Y=labelencoder_y.fit_transform(Y)
#splitting the data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
