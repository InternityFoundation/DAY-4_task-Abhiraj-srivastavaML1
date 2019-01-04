import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
data=pd.read_csv('auto-mpg.csv')
data=data.replace('?',np.NaN)
X=data.iloc[:,1:8].values
Y=data.iloc[:,0].values
from sklearn.preprocessing import Imputer 
imputer=Imputer(missing_values='NaN',strategy='mean',axis=1)
imputer.fit(X[:,2].reshape(1,-1))
X[:,2]=imputer.transform(X[:,2].reshape(1,-1))
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_predict=regressor.predict(X_test)
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Specification')
plt.ylabel('mpg')