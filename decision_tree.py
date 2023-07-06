'''Write a classification program for implementing decision tree using
pima-indians-diabetes dataset.'''

import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("diabetes.csv")
feature_cols = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = df[feature_cols]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy)
classrep = classification_report(y_test, y_pred)
print(classrep)
