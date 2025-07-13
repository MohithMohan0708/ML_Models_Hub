import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the 'tips' dataset
data = sns.load_dataset('tips')

# Define features and target
X = data[['total_bill', 'tip']]
y = data['sex'].apply(lambda x: 1 if x == "Male" else 0)  # Convert target to binary: 1 for Male, 0 for Female

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the Decision Tree model
model_dt = DecisionTreeClassifier(random_state=0)
model_dt.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = model_dt.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Classification Accuracy: {accuracy:.2f}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_dt)
print('Confusion Matrix:')
print(conf_matrix)

# Generate classification report
class_report = classification_report(y_test, y_pred_dt)
print('Classification Report:')
print(class_report)

# Plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model_dt, feature_names=['total_bill', 'tip'], class_names=['Female', 'Male'], filled=True, rounded=True)
plt.title('Decision Tree Structure')
plt.show()
