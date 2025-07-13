import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris(as_frame=True).frame


# Define features and target (Binary classification: Class 0 vs others)
X = data[['sepal length (cm)', 'sepal width (cm)']]
y = (data['target'] == 0).astype(int)  # Convert to binary classification

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the SVM model
model_svm = SVC(kernel='linear')
model_svm.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = model_svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_svm)
print(f'SVM Classification Accuracy: {accuracy:.2f}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_svm)
print('Confusion Matrix:')
print(conf_matrix)

# Generate classification report
class_report = classification_report(y_test, y_pred_svm)
print('Classification Report:')
print(class_report)
