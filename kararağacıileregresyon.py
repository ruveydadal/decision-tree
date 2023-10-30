# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:22:32 2021

@author: muham
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler=pd.read_csv('maaslar.csv')

#verilerin bölünmesi
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y)
plt.plot(x,lin_reg.predict(x))

#polynomial regression(2.dereceden)
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()

#polynomial regression(4.dereceden)
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()

#decision tree regression
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,r_dt.predict(x),color='blue')

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))









