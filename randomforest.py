# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:31:16 2020

@author: Pyxis
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import math
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Displaying files
import os
os.chdir("C:\\Users\\Pyxis\\Documents\\Arjun's Docs\\Kaggle\Titanic ML Dataset")
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Reading train.csv
data = pd.read_csv("train.csv")
meanAge = data.mean(axis = 0, skipna = True)["Age"];
ageColId = data.columns.get_loc("Age");

for i in range(len(data)):
    if np.isnan(data.loc[i]["Age"]):
        data.iloc[i,ageColId] = meanAge;

# Binning 10 age groups
minAge = data.min(axis = 0, skipna = True)["Age"];
maxAge = data.max(axis = 0, skipna = True)["Age"];

data['AgeGroup'] = pd.qcut(data['Age'], q=7);

ordered_id_list = []
for e in range(len(data)):
    ordered_id_list.append(e)
random.shuffle(ordered_id_list);

train_breakpoint = math.ceil(0.7*len(data));

#train_data = data.loc[:train_breakpoint];
#test_data = data.loc[train_breakpoint+1:];

train_data = pd.DataFrame();
test_data = pd.DataFrame();

for i in range(train_breakpoint):
    train_data = train_data.append(data.loc[ordered_id_list[i]],ignore_index=True)
for i in range(train_breakpoint,len(data)):
    test_data = test_data.append(data.loc[ordered_id_list[i]],ignore_index=True)
    
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]#, "Fare", "AgeGroup"];# , "Age"

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)

corrects = 0;

for i in range(len(test_data)):
    if (predictions[i] == test_data.loc[i]["Survived"]):
        corrects += 1;
        
accuracy = corrects/len(test_data);
print("Accuracy is " + str(accuracy) + ".");
    
    