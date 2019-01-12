# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 22:32:38 2019

@author: SP Srivastava
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('audit_risk.csv')
X=data.iloc[:,2:26].values
Y=data.iloc[:,-1].values
from sklearn.preprocessing import Imputer 
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer.fit(X)
X=imputer.transform(X)
from sklearn.cross_validation import train_test_split,cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
mylist=list(range(1,50))
neighbors=list(filter(lambda x:x%2!=0,mylist))
cv_scores=[]
for k in neighbors:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    cv_scores.append(scores.mean())
MSE=[1-x for x in cv_scores]
optimal_k=neighbors[MSE.index(min(MSE))]
print(optimal_k)
classifier=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_predict=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_predict)
print(classification_report(y_test,y_predict))
