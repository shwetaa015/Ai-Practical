import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('iris.csv')
x = data.drop('Target', axis = 1)
y = data['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

svm_classifier = SVC(kernel = 'linear')
svm_classifier.fit(x_train, y_train)
y_pred = svm_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
