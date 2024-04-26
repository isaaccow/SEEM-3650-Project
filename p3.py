import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from Excel file
data = pd.read_excel('data.xlsx')

# Extract columns related to greenhouse gas emissions, temperature, and carbon intensity
emissions = data['Total GHG Emissions']
temperature = data['Average Temperature']
carbon_intensity = data['Carbon Intensity(per HK Dollar GDP)']

# Create a DataFrame with the variables of interest
df = pd.DataFrame({'Emissions': emissions, 'Temperature': temperature, 'Carbon Intensity': carbon_intensity})

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
