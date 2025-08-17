from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.2,random_state=77,stratify=y)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_grid = {
    "C" : [0.1, 1, 10, 100],
    "gamma" : [0.1, 1, 10, 100],
    "kernel" : ['linear', 'rbf']
}
grid = GridSearchCV(SVC(class_weight='balanced'),param_grid,cv=5,scoring='accuracy')
grid.fit(x_train_std, y_train)
print("Best parameters found: ", grid.best_params_)
print("Best cross-validation score: ", grid.best_score_)
# svm = SVC(kernel='linear',random_state=77,C=0.1,gamma='auto')
# svm.fit(x_train_std,y_train)
# from sklearn.metrics import accuracy_score
# y_pred = svm.predict(x_test_std)
# print(accuracy_score(y_test,y_pred))
