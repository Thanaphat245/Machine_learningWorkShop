from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=500,n_features=13,n_classes=2,random_state=77)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=77)
model = MLPClassifier(hidden_layer_sizes=(100,50,20,10),activation='relu',solver='adam',random_state=77)
from sklearn.metrics import accuracy_score
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = accuracy_score(y_test,y_pred)
print(mse)