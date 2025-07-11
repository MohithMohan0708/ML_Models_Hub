import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('synthetic_dataset.csv')
X = data[['Income']].values
y = data['Purchase Amount'].values

model_ols = LinearRegression()
model_ols.fit(X,y)

y_pred_ols = model_ols.predict(X)
plt.scatter(data['Income'],data['Purchase Amount'],color='blue')
plt.plot(data['Income'],y_pred_ols,color='purple')
plt.xlabel('Income')
plt.ylabel('Purchase Amount')
plt.title('Linear Regression using OLS')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

data = pd.read_csv('synthetic_dataset.csv')
X = data[['Income']].values
y = data['Purchase Amount'].values

model_gd = SGDRegressor(learning_rate='constant',max_iter=1000)
model_gd.fit(X,y)

y_pred_gd = model_gd.predict(X)
plt.scatter(data['Income'],data['Purchase Amount'],color='blue')
plt.plot(data['Income'],y_pred_gd,color='red')
plt.xlabel('Income')
plt.ylabel('Purchase Amount')
plt.title('Linear Regression using GD')
plt.show()

