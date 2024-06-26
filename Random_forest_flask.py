# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:24:41 2024

@author: VEDSD
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pickle


#%%
#importing the data
data=pd.read_csv('D:\MSc(ASA)\Python\Weather Data.csv')
data.isnull().sum()

data.columns=['Temperature','Dew','Humidity','Wind_Speed','Visibility_km','Pressure','Weather']
#%%
#Since we are using the same data and there wasn't any missing values, we can directly go for the model fitting
#We are using Random Forest for the classification of our data.
len(data["Weather"].unique())
#%%
RF_model=RandomForestClassifier(n_estimators=1000, random_state=50)
RF_model.fit(data.drop('Weather', axis=1), data["Weather"])

#%%
#CrossValidation
# Define features (X) and target (y)
X = data.drop('Weather', axis=1)
y = data['Weather']

# Initialize the Random Forest classifier
model_obj = RandomForestClassifier(n_estimators=1000, random_state=50)

# Initialize StratifiedKFold with 5 folds
cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)

# List to store accuracy scores for each fold
accuracy_scores = []

# Perform cross-validation
for train_index, test_index in cross_val.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train the classifier
    model_obj.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model_obj.predict(X_test)
    
    # Calculate accuracy and store it
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Print accuracy scores for each fold
for i, accuracy in enumerate(accuracy_scores):
    print(f"Fold {i+1} Accuracy: {accuracy}")

# Print mean and standard deviation of accuracy scores
print(f"Mean Accuracy: {np.mean(accuracy_scores)}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_scores)}")


#%%
pickle.dump(RF_model, open('Flask_RF.pkl','wb'))
model=pickle.load(open('Flask_RF.pkl','rb'))