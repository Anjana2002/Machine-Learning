import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import seaborn as sns

digits = pd.read_csv("mnist_test.csv")
X = digits.iloc[:,1:]
y = digits.iloc[:,0]
X=scale(X)
X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.2,random_state=43)
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy :{accuracy}")
class_report = classification_report(y_test, y_pred )
print(class_report)

four = digits.iloc[3,1:]
four = four.values.reshape(28, 28)
plt.imshow(four, cmap='gray')
plt.show()
