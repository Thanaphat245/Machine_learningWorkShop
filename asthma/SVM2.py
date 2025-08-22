import pandas as pd
data = pd.read_csv('asthma/asthma_disease_data.csv')
data = data.drop(['DoctorInCharge', 'PatientID'], axis=1)
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import SMOTE
smote_enn = SMOTE(random_state=77)
X_resampled, y_resampled = smote_enn.fit_resample(X_train_sc, y_train)
print("Resampled X shape:", X_resampled.shape)
print("Resampled y shape:", y_resampled.sum())
from sklearn.svm import SVC

model = SVC(random_state=77, kernel='rbf', C=10, gamma=0.1,class_weight='balanced') 
model.fit(X_resampled, y_resampled) 

from sklearn.metrics import classification_report, confusion_matrix 
y_pred = model.predict(X_test_sc) 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))