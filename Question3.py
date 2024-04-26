# The decision tree regression model is used

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from Excel file
data = pd.read_excel('data.xlsx')

# Extract the relevant columns
X = data[['Total GHG Emissions', 'Average Temperature']]
y = data['Sea Level Rise at Victoria Harbour']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree regressor
model = DecisionTreeRegressor()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)
print("Coefficient of Determination (R-squared):", r2)

# Plotting the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_train['Total GHG Emissions'], X_train['Average Temperature'], y_train, c='blue', label='Actual Train')
ax.scatter(X_test['Total GHG Emissions'], X_test['Average Temperature'], y_test, c='green', label='Actual Test')
ax.scatter(X_test['Total GHG Emissions'], X_test['Average Temperature'], y_test_pred, c='red', label='Predicted')

ax.set_xlabel('Total GHG Emissions')
ax.set_ylabel('Average Temperature')
ax.set_zlabel('Sea Level Rise')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()
