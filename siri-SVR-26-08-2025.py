# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:18:12 2025

@author: ttwrd
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\ttwrd\OneDrive\Attachments\Desktop\emp_sal.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# linear model  -- linear algor ( degree - 1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred
# polynomial model  
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(x)

poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

from sklearn.svm import SVR
#svr_reg=SVR(kernel='sigmoid',degree=2,gamma='scale',C=10.0 )
svr_reg=SVR(kernel='poly',degree=4,gamma='scale',C=10.0 )
#degree 4 gamma scale is best prediction model 

svr_reg.fit(x,y)

svr_model_pred=svr_reg.predict([[6.5]])
print(svr_model_pred)
