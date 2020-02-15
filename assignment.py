# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:38:38 2020

@author: kiran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_excel('mavoix_ml_sample_dataset.xlsx')
data = data.drop(data.columns[10], axis=1)
data.hist()
data.plot(kind='box',subplots=True,layout=(5,5),sharex=False,sharey=False)
X = data.iloc[:,2:15]
y = data.iloc[:,0]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =3)
knn.fit(X,y)
knn.predict([[3,3,2,3,2,3,0,0,0,0,0,0,0],[2,3,3,0,0,2,0,0,0,0,0,0,0],[2,2,3,2,0,2,0,0,0,0,0,0,0],[2,2,3,2,0,2,2,0,0,0,0,0,0],[0,0,3,2,2,2,3,0,0,0,0,0,0],[3,0,2,3,3,2,3,0,0,0,0,0,0],[0,0,0,2,2,2,3,0,0,0,0,3,3],[0,0,3,2,2,2,3,2,2,2,0,0,0],[2,0,3,2,2,2,2,2,3,0,0,0,0],[3,3,3,0,0,2,0,0,0,0,0,0,0]])

 



 
    
 


































X = dataset.iloc[:, 2:15].values

y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)

 '''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

'''


New=knn.predict([[2,0,0,1,0,3,2,1,0,0,0,0]])
print(New)

   