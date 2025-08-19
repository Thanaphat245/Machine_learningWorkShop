import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Component skcikit

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target,name = "MedHouseVal")

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=77)

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=77,criterion='poisson',max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))   
# 3. วาดกราฟเปรียบเทียบค่าจริงกับค่าที่ทำนาย
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red', linestyle='--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()
# แสดงภาพต้นไม้ตัดสินใจ
plt.figure(figsize=(20, 10))
from sklearn.tree import plot_tree
plot_tree(model,feature_names=housing.feature_names, filled=True, fontsize=10)
plt.title("Decision Tree Regressor")
plt.show()