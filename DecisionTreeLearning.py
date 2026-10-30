import pandas
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pandas.read_csv('iris.csv')
x = data.drop('Target', axis = 1)
y = data['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
plt.figure(figsize =(12, 8))
plot_tree(clf, filled = True, feature_names = x.columns, class_names = data['Target'].unique())
plt.title("Decision Tree Visualization")
plt.show()
