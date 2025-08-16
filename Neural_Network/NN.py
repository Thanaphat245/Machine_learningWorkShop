from sklearn.datasets import load_breast_cancer
lod = load_breast_cancer()
X = lod.data
y = lod.target
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(32,32,32,32,32,32),max_iter=1000,random_state=77,activation="relu",solver="adam")
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_curve
accuracy_score = accuracy_score(y_test,y_pred)
f1_score_micro = f1_score(y_test,y_pred,average ='micro')
f1_score_macro = f1_score(y_test,y_pred,average ='macro')
confusion_matrix=confusion_matrix(y_test,y_pred)
roc_curve = roc_curve(y_test,y_pred)
print("accuracy_score = ",format(accuracy_score,",.2f"))
print("f1_score_macro = ",format(f1_score_macro,",.2f"))
print("f1_score_micro = ",format(f1_score_micro,",.2f"))
print("confusion_matrix = ",confusion_matrix)
# print("roc_curve = ",roc_curve)