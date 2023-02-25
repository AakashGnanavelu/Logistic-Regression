# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 13:28:39 2021

@author: Aakash
"""
import numpy as np
import pandas as pd

raw_data = pd.read_csv(r"C:\Users\Aakash\Desktop\AAKASH\Coding Stuff\Full Data Science\Logistic Regression\Assginments\bank.csv")

input = []

for x in range(0, len(raw_data)):
    row = raw_data.iloc[x]
    row = list(row)
    final = row[0]
    final = final.split(";")
    input.append(final)
    
data = pd.DataFrame(input)
data.columns = ['age',"job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
data_copy = data.copy()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labelencoder = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')

data['job'] = labelencoder.fit_transform(data['job'])
job_df = pd.DataFrame(enc.fit_transform(data[['job']]).toarray())

data['marital'] = labelencoder.fit_transform(data['marital'])
marital_df = pd.DataFrame(enc.fit_transform(data[['marital']]).toarray())

data['education'] = labelencoder.fit_transform(data['education'])
education_df = pd.DataFrame(enc.fit_transform(data[['education']]).toarray())

data['contact'] = labelencoder.fit_transform(data['contact'])
contact_df = pd.DataFrame(enc.fit_transform(data[['contact']]).toarray())

data['poutcome'] = labelencoder.fit_transform(data['poutcome'])
poutcome_df = pd.DataFrame(enc.fit_transform(data[['poutcome']]).toarray())

marital_df.columns = ['divorced','married','single']
education_df.columns = ['primary','secondary','tri', 'quad']
contact_df.columns = ['cellular','telephone','unknown']
poutcome_df.columns = ['a','b','c','d']

housing_dict = {'housing': {'"yes"':1, '"no"' :0}}
loan_dict = {'loan': {'"yes"':1, '"no"' :0}}
default_dict = {'default': {'"yes"':1, '"no"' :0}}
y_dict = {'y': {'"yes"':1, '"no"' :0}}

data = data.replace(housing_dict)
data = data.replace(loan_dict)
data = data.replace(y_dict)
data = data.replace(default_dict)

del data['job']
del data['marital']
del data['education']
del data['contact']
del data['month']
del data['poutcome']

data = data.join(job_df)
data = data.join(marital_df)
data = data.join(education_df)
data = data.join(contact_df)
data = data.join(poutcome_df)

X = data.copy()
del X['y']
Y = data.iloc[:,10]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(random_state = 0)
logit.fit(X_train,Y_train)

predict_train = logit.predict(X_train)
predict_test = logit.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cnf_test_matrix = confusion_matrix(Y_test, predict_test > 0.5 )
cnf_test_matrix

print(accuracy_score(Y_test, predict_test > 0.5))

cnf_train_matrix = confusion_matrix(Y_train, predict_train > 0.5 )
cnf_train_matrix

print(accuracy_score(Y_train, predict_train > 0.5))
