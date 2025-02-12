# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:14:49 2024

@author: Ishani
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

year_2007 = pd.read_csv('D:/Research Assistant/CSV_Dataset/2007.csv')
year_2007.head()
print("Data is imported.")

#Creating function to convert arrival delay in binary 

def ArrDelay_Yes(row):
    if row['ArrDelay'] > 0:
        return 1  # Delayed
    else:
        return 0  # On time

year_2007['ArrDelay_Binary'] = year_2007.apply(ArrDelay_Yes, axis=1) #axis =0 is rows and axis=1 is columns
print(year_2007['ArrDelay_Binary'])

print("ArrDelay_Binary column created.")

#Creating function to identify Time Period

year_2007['ArrTime_Trim']= year_2007['ArrTime'].astype(str).str.strip('.0')

def categorize_time_period(hour):
  try:
    hour = int(hour)
    if 6 <= hour < 12:
      return 'Morning'
    elif 12 <= hour < 18:
      return 'Afternoon'
    elif 18 <= hour < 21:
      return 'Evening'
    else:
      return 'Night'
  except Exception as ex:
    pass

#Creating Tine Period Column 
temp = []
for i in year_2007['ArrTime_Trim'].to_list():
  if len(str(i)) == 3:
    temp.append(categorize_time_period(i[:1]))
  elif len(str(i)) == 4:
    temp.append(categorize_time_period(i[:2]))

temp

temp1=pd.DataFrame(temp,columns=['DepCategory'])
year_2007['TimePeriod']=temp1
year_2007['TimePeriod'].dropna(inplace=True)

print("Year 2007 ArrCategory created.")

#Defining features

features = ['DayOfWeek', 'Month', 'Distance', 'TimePeriod']
X = year_2007[features]
y = year_2007['ArrDelay_Binary']

#Handling Categorical variables
X = pd.get_dummies(X, columns=['DayOfWeek', 'Month', 'TimePeriod'])

# Handling missing values
imputer = SimpleImputer(strategy='mean')  # Replace with your preferred strategy
X_imputed = imputer.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Creating and trainning the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix)
print("Classification Report:\n", classification_report)

