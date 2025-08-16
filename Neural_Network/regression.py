from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
X,y= make_regression(n_samples=200,random_state=77,n_features=13,noise=0.1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=77,test_size=0.2)
model = MLPRegressor(hidden_layer_sizes=(110,55,35,15),activation='relu',random_state=77,max_iter=10000,solver='adam')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
print(mse)