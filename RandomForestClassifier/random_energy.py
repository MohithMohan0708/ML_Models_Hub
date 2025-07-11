import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Energy_dataset.csv')

# Define a single feature and binary target (e.g., Heating Load threshold of 15)
X = data[['Heating Load']]  # Single feature
y = (data['Cooling Load'] > 15).astype(int)  

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=0)
model_rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = model_rf.predict(X_test)
y_pred_proba_rf = model_rf.predict_proba(X_test)[:,1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy:.2f}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
print('Confusion Matrix:')
print(conf_matrix)

# Calculate F1 Score
f1 = f1_score(y_test, y_pred_rf)
print(f'F1 Score: {f1:.2f}')

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba_rf)
print(f'ROC AUC Score: {roc_auc:.2f}')

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
