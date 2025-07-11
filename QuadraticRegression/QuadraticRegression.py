import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('synthetic_dataset.csv')

X = data[['Income']].values
y = data['Purchase Amount'].values

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model_quad = LinearRegression()
model_quad.fit(X_poly,y)

y_pred_quad = model_quad.predict(X_poly)

plt.scatter(data['Income'],data['Purchase Amount'],color='blue')
plt.plot(data['Income'],y_pred_quad,color='red')
plt.xlabel('Income')
plt.ylabel('Purchase Amount')
plt.title('Quadratic Regression')
plt.show()