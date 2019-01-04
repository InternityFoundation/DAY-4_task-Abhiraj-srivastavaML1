import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('Position_Salaries.csv')
X=data.iloc[:,1:2].values
Y=data.iloc[:,2].values

"""from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=1/3,random_state=0)"""
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X,Y)
y_predict1=lin.predict(X)
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=9)
X_poly=poly.fit_transform(X)
lin2=LinearRegression()
lin2.fit(X_poly,Y)
y_predict2=lin2.predict(X_poly)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin2.predict(poly.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
