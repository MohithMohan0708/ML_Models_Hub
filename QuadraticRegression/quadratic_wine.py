import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_wine

# Load the Wine dataset
data = load_wine(as_frame=True).frame
X = data[['malic_acid']].values
y = data['alcohol'].values


# Transform to polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Quadratic Regression
model_quad = LinearRegression()
model_quad.fit(X_poly, y)

y_pred_quad = model_quad.predict(X_poly)

# Plot results
plt.scatter(data['malic_acid'], data['alcohol'], color='blue', label='Actual Data')
plt.plot(data['malic_acid'], y_pred_quad, color='red', label='Quadratic Prediction')
plt.xlabel('Malic Acid')
plt.ylabel('Alcohol')
plt.title('Quadratic Regression')
plt.legend()
plt.show()
