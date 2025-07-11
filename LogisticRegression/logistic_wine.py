import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Load the Wine dataset
data = load_wine(as_frame=True).frame
X = data[['malic_acid']].values
y = (data['target'] == 1).astype(int)  # Binary classification: Class 1 vs others

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Logistic Regression model
model_log = LogisticRegression()
model_log.fit(X_train, y_train)

y_pred_log = model_log.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred_log)
print(f"Accuracy : {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_log)
print("Confusion Matrix: ")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred_log)
print("Classification Report: ")
print(class_report)
