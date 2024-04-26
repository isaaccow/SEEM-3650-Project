# linear regression model is used

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read data from Excel file
data = pd.read_excel('data.xlsx')

# Extract columns related to greenhouse gas emissions, temperature, and carbon intensity
emissions = data['Total GHG Emissions']
temperature = data['Average Temperature']
carbon_intensity = data['Carbon Intensity(per HK Dollar GDP)']

# Prepare the data for multiple linear regression
X = pd.DataFrame({'Emissions': emissions, 'Temperature': temperature})
y = carbon_intensity

# Create separate linear regression models for each feature
model_emissions = LinearRegression()
model_temperature = LinearRegression()

# Fit the models to the data
model_emissions.fit(X[['Emissions']], y)
model_temperature.fit(X[['Temperature']], y)

# Make predictions using the linear regression models
y_pred_emissions = model_emissions.predict(X[['Emissions']])
y_pred_temperature = model_temperature.predict(X[['Temperature']])

# Calculate the coefficient of determination (R-squared) as a performance metric
r2_emissions = r2_score(y, y_pred_emissions)
r2_temperature = r2_score(y, y_pred_temperature)

# Plot the actual data and the predicted data for 'Total GHG Emissions'
plt.scatter(emissions, carbon_intensity, color='blue', label='Actual Carbon Intensity')
plt.plot(emissions, y_pred_emissions, color='red', label='Predicted Carbon Intensity')
plt.xlabel('Total GHG Emissions')
plt.ylabel('Carbon Intensity')
plt.title('Total GHG Emissions vs. Carbon Intensity')
plt.legend()
plt.show()

# Plot the actual data and the predicted data for 'Average Temperature'
plt.scatter(temperature, carbon_intensity, color='blue', label='Actual Carbon Intensity')
plt.plot(temperature, y_pred_temperature, color='red', label='Predicted Carbon Intensity')
plt.xlabel('Average Temperature')
plt.ylabel('Carbon Intensity')
plt.title('Average Temperature vs. Carbon Intensity')
plt.legend()
plt.show()

print(f"Coefficient of Determination for Total GHG Emissions (R-squared): {r2_emissions}")
print(f"Coefficient of Determination for Average Temperature (R-squared): {r2_temperature}")
