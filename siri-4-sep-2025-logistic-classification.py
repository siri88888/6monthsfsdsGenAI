# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 07:57:52 2025

@author: Siri
"""

# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORT DATASET 
#dataset = pd.read_csv(r"C:\Users\A3MAX SOFTWARE TECH\Desktop\WORK\1. KODI WORK\1. NARESH\1. MORNING BATCH\N_Batch -- 8.00AM\3. APR\8th,12th - logit\2.LOGISTIC REGRESSION CODE\logit classification.csv")
dataset=pd.read_csv(r"C:\Users\ttwrd\Downloads\logit classification.csv")

# SPLIT WITH IV & DV
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# 75-25 TRAIN TEST SPLIT 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=0)

# SCALE THE DATA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# TRAIN THE MODEL X_TRAIN & Y_TRAIN 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# create prediction for x_test 
y_pred = classifier.predict(X_test)

# confusion matrix now 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# MODEL ACCURACY
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

#TRAINING ACCURACY
bias = classifier.score(X_train,y_train)
print(bias)

# TESTING ACCURACY
variance = classifier.score(X_test, y_test)
print(variance)



#04-september-2025   Logistic clasiification 

dataset1=pd.read_csv(r"C:\Users\ttwrd\Downloads\logit classification.csv")
d1=dataset1.copy()
# IMPORT LIBRARIES

# SPLIT WITH IV & DV
dataset1= dataset1.iloc[:, [3, 4]].values

# SCALE THE DATA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
M=sc.fit_transform(dataset1)

M
y_pred1=pd.DataFrame()

d1['ypred1']=classifier.predict(M)
d1.to_csv('final.csv')