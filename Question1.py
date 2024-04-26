import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_excel('data.xlsx')

emissions = data['Total GHG Emissions']
temperature = data['Average Temperature']

X = emissions.values.reshape(-1, 1)
y = temperature.values

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Linear Regression')
plt.xlabel('Total GHG Emissions')
plt.ylabel('Average Temperature')
plt.title('Relationship between GHG Emissions and Temperature')
plt.legend()
plt.show()

