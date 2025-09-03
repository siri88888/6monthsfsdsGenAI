# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print('hai')
import numpy as np
import matplotlib as plt
import seaborn as sns
import pandas as pd

dataset = pd.read_csv(r"C:\Users\ttwrd\Downloads\logit classification.csv")


X = dataset.iloc[:, [2.3]].values	
#
y = dataset.iloc[:,-1].values  

from sklearn.model_selection   import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=51)#tested for 0,100

#with scaling
from sklearn.preprocessing  import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


"""#with Normalization

from sklearn.preprocessing  import Normalizer
sc=Normalizer()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)"""

from sklearn.linear_model  import LogisticRegression
#classifier=LogisticRegression()
classifier=LogisticRegression(solver="lbfgs")

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics  import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

print(cm)

from sklearn.metrics  import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac) 

from sklearn.metrics  import  classification_report
cr = classification_report(y_test, y_pred)
print(cr)

bias=classifier.score(X_train, y_train)
print(bias)
variance=classifier.score(X_test,y_test)
print(variance)





