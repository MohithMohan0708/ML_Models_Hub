import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv('synthetic_dataset.csv')
X = data[['Study Hours']].values
y = data['Pass/Fail Outcome'].values

X_test,X_train,y_test,y_train = train_test_split(X,y,random_state=0,test_size=0.2)

model_log = LogisticRegression()
model_log.fit(X_train,y_train)

y_pred_log = model_log.predict(X_test)

accuracy = accuracy_score(y_test,y_pred_log)
print(f"Accuracy : {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test,y_pred_log)
print("Confusion Matrix: ")
print(conf_matrix)

class_report = classification_report(y_test,y_pred_log)
print("Classification Report: ")
print(class_report)
