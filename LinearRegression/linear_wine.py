import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_wine

# Load the Wine dataset
data = load_wine(as_frame=True).frame
X = data[['malic_acid']].values
y = data['alcohol'].values


# OLS Linear Regression
model_ols = LinearRegression()
model_ols.fit(X, y)
y_pred_ols = model_ols.predict(X)

# Plot results
plt.scatter(data['malic_acid'], data['alcohol'], color='blue', label='Actual Data')
plt.plot(data['malic_acid'], y_pred_ols, color='red', label='OLS Prediction')
plt.xlabel('Malic Acid')
plt.ylabel('Alcohol')
plt.title('Linear Regression (OLS)')
plt.legend()
plt.show()

# Gradient Descent Regression
from sklearn.linear_model import SGDRegressor


# SGD Regressor
model_gd = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000)
model_gd.fit(X, y)
y_pred_gd = model_gd.predict(X)

# Plot results
plt.scatter(data['malic_acid'], data['alcohol'], color='blue', label='Actual Data')
plt.plot(data['malic_acid'], y_pred_gd, color='green', label='GD Prediction')
plt.xlabel('Malic Acid')
plt.ylabel('Alcohol')
plt.title('Linear Regression (Gradient Descent)')
plt.legend()
plt.show()
