from sklearn.datasets import load_wine
wine = load_wine()
X = wine.data
y = wine.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77,)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini',random_state=77,splitter='best',max_depth=4)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
classification_report = classification_report(y_test, y_pred)
print(classification_report)
print(confusion_matrix(y_test, y_pred))

# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree
# plt.figure(figsize=(20,10))
# plot_tree(model, filled=True)
# plt.show()