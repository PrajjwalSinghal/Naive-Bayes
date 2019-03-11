#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:32:23 2019

@author: prajjwalsinghal
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
def getAccuracy(Y_test, Y_Pred):
    correct = 0;
    for x in range(len(Y_test)):
        if Y_test.iloc[x] == Y_Pred[x]:
            correct += 1
    return (correct/float(len(Y_test))) * 100.0


# Importing the dataset
dataset = pd.read_csv('Iris.csv')
dataset = dataset.drop('Id', axis = 1)


# Encoding the categorical data
dataset["Species"]=np.where(dataset["Species"]=="Iris-setosa",0,
                                  np.where(dataset["Species"]=="Iris-versicolor",1,
                                           np.where(dataset["Species"]=="Iris-virginica",2,3)
                                          ))

# Splitting the dataset into training and test set

X_train, X_test = train_test_split(dataset, test_size = 0.3, random_state = 0)

gnb = GaussianNB()
used_features = [
        "SepalLengthCm", 
        "SepalWidthCm", 
        "PetalLengthCm", 
        "PetalWidthCm"
        ]
gnb.fit(X_train[used_features].values,
        X_train["Species"])

Y_pred = gnb.predict(X_test[used_features])

print(getAccuracy(X_test['Species'], Y_pred))

#   Accuracy = 100%

