'''Write a classification program for implementing KNN.'''

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as myplot

df = pd.read_csv("diabetes.csv")
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.4,random_state=42)
neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train,y_train)

test_accuracy[i] = knn.score(X_test, y_test)
myplot.title('k-NN Varying number of neighbors')
myplot.plot(neighbors, test_accuracy, label='Testing Accuracy')
myplot.plot(neighbors, train_accuracy, label='Training accuracy')
myplot.legend()
myplot.xlabel('Number of neighbors')
myplot.ylabel('Accuracy')
myplot.show()

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
class_report = classification_report(y_test, y_pred )
print(class_report)
