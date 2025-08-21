from sklearn.ensemble import RandomForestClassifier
import pandas as pd
df = pd.read_csv('asthma/asthma_disease_data.csv')
data = df.drop(['DoctorInCharge','PatientID'],axis=1)
X = data.drop('Diagnosis',axis = 1)
y = data['Diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from imblearn.over_sampling import SMOTE    
sm = SMOTE(random_state=42)
X_res,Y_res = sm.fit_resample(X_train, y_train)
print("Before SMOTE:", X_train.shape, y_train.shape)
print("After SMOTE:", X_res.shape, Y_res.shape)

model = RandomForestClassifier(n_estimators=100, random_state=42,min_samples_split=2)

model.fit(X_res,Y_res)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))