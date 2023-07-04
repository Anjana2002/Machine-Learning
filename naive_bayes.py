
'''Write a classification program for implementing Naïve Bayes algorithm using iris dataset'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.datasets import load_iris

iris = load_iris()
X_train,X_test, y_train, y_test =train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:")
print(accuracy)
class_report=classification_report(y_test,y_pred)
print("Classification Report:")
print(class_report)
