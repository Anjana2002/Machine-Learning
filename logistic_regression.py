'''Write a classification program for implementing logistic regression using wine dataset'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

wine = pd.read_csv("wine.csv")
#Handle missing values
for col in wine.columns:
     if wine[col].isnull().sum()>0:
         wine[col] = wine[col].fillna(wine[col].mean())
#Check if there are any remaining missing values
wine.isnull().sum().sum()
wine.replace({'white':1,'red':0}, inplace=True)

#split the dataset into feature (X) and target (y)
X = wine.drop('quality', axis = 1)
y = wine['quality']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
#Create a logistic regression model
logreg = LogisticRegression()
#fit it to the training data
logreg.fit(X_train, y_train)
#Predict the target variable
y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("ACCURACY:")
print(accuracy)

classreport = classification_report(y_test, y_pred)
print("Classification Report")
print(classreport)
