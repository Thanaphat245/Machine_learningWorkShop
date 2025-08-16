import pandas as pd

df = pd.read_csv("C:/Users/Thanaphat/Desktop/ML/archive/heart_failure_clinical_records_dataset.csv")

# print(df.head())
#print(df.info())

from sklearn.model_selection import train_test_split
x = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(x)

# plt.figure(figsize=(8,6))
# plt.scatter(X_pca[:,0],X_pca[:,1],c = y,cmap="coolwarm",edgecolors='k')
# plt.xlabel("PCA Component 1")      # ป้ายแกนนอน
# plt.ylabel("PCA Component 2")      # ป้ายแกนตั้ง
# plt.title("PCA Visualization")     # หัวข้อกราฟ
# plt.colorbar(label="Class")        # แถบสีสำหรับ class (target)
# plt.show()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=77)

from sklearn.svm import SVC

model = SVC(kernel='linear',random_state=77,class_weight='balanced')

model.fit(x_train,y_train)

y_predic = model.predict(x_test)

from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,f1_score
cm = confusion_matrix(y_test,y_predic)
rs = recall_score(y_test,y_predic)
ps = precision_score(y_test,y_predic)
ac = accuracy_score(y_test,y_predic)
f1s = f1_score(y_test,y_predic)
# print("confusion_matrix",cm)
# print("recall_score",rs)
# print("precision_score",ps)
# print("accuracy_score",ac)
# print("f1_score",f1s)

# การ run เช็ค model kernel ทุกตัว 
from sklearn.model_selection import cross_val_score

for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    model = SVC(kernel=kernel, class_weight='balanced')
    scores = cross_val_score(model, x, y, cv=5, scoring='f1')
    print(f"{kernel}: f1={scores.mean():.3f}")

