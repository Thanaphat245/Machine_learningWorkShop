from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split    

iris = load_iris()
X,y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)
model = DecisionTreeClassifier(random_state=77,criterion='gini',splitter='best')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# print(model.score(X, y))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, fontsize=10)
plt.title("Decision Tree Classifier")
plt.show()  