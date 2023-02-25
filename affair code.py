# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:59:53 2021

@author: Aakash
"""

import pandas as pd

data = pd.read_csv(r"C:\Users\Aakash\Desktop\AAKASH\Coding Stuff\Full Data Science\Logistic Regression\Assginments\affair.csv")

data.columns = ['index','n_affairs','kids','vryunhap','unhap','avgmarr','hapavg','vryhap','antirel',
                'notrel','slghtrel','smerel','vryrel','yrsmarr1','yrsmarr2','yrsmarr3','yrsmarr4',
                'yrsmarr5','yrsmarr6']

del data['index']

for x in range (0,len(data['n_affairs'])):
    if data['n_affairs'][x] > 0:
        data['n_affairs'][x] = 1

data = data.dropna()

data.describe()
data.head()

from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(data, test_size = 0.3)

import statsmodels.formula.api as sm
logit_model = sm.logit('n_affairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel+ yrsmarr1 + yrsmarr2+ yrsmarr3+ yrsmarr4+ yrsmarr5+ yrsmarr6', data = train_data).fit()
logit_model.summary()

predict_test = logit_model.predict(pd.DataFrame(test_data[['n_affairs','kids','vryunhap','unhap','avgmarr','hapavg','vryhap','antirel','notrel','slghtrel','smerel','vryrel','yrsmarr1','yrsmarr2','yrsmarr3','yrsmarr4','yrsmarr5','yrsmarr6']]))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cnf_test_matrix = confusion_matrix(test_data['n_affairs'], predict_test > 0.5 )
cnf_test_matrix

print(accuracy_score(test_data.n_affairs, predict_test > 0.5))

predict_train = logit_model.predict(pd.DataFrame(train_data[['n_affairs','kids','vryunhap','unhap','avgmarr','hapavg','vryhap','antirel','notrel','slghtrel','smerel','vryrel','yrsmarr1','yrsmarr2','yrsmarr3','yrsmarr4','yrsmarr5','yrsmarr6']]))

cnf_train_matrix = confusion_matrix(train_data['n_affairs'], predict_train > 0.5 )
cnf_train_matrix

print(accuracy_score(train_data.n_affairs, predict_train > 0.5))
