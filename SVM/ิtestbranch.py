from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=77)
from sklearn.svm import SVC
svm =SVC(kernel='rbf',random_state=77)
svm.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,mean_squared_error
y_predic = svm.predict(x_test)
mse = mean_squared_error(y_test,y_predic)
print(mse)
for i in range(len(y_predic)):
    print(i+1,y_test,y_predic[i])
from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,f1_score,ConfusionMatrixDisplay
confusion_M = confusion_matrix(y_test,y_predic)
recall_sc = recall_score(y_test,y_predic)
precision_s = precision_score(y_test,y_predic)
accuracy_s = accuracy_score(y_test,y_predic)
f1_s = f1_score(y_test,y_predic)
print("confusion_matrix = ",confusion_M)
print("recall_score = ",recall_sc)
print("precision_score = ",precision_s)
print("accuracy_score = ",accuracy_s)
print("f1_score",f1_s)

import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_M,
                              display_labels=svm.classes_)
disp.plot()
plt.show()
