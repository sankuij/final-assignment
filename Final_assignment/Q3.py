import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

predict_2024 = pd.read_csv ("finalproject/3./ATBe-6.csv")

# Filter data for relevant columns: technology, year, and cost (value)
predict_2024 = predict_2024[predict_2024['scenario'] == 'Moderate']
predict_2024 = predict_2024[predict_2024['core_metric_case'] == 'R&D']
predict_2024 = predict_2024[predict_2024['core_metric_parameter'] == 'LCOE']

filtered_data = predict_2024[['core_metric_variable', 'technology_alias', 'value']]

filtered_data = filtered_data.groupby(['core_metric_variable', 'technology_alias'], as_index=False).mean()

filtered_data.rename(columns={'core_metric_variable': 'Year', 'technology_alias': 'Technology'}, inplace=True)

pivoted_data = filtered_data.pivot(index='Year', columns='Technology', values='value')

slopes = {}

for technology in pivoted_data.columns:
    # Prepare the data
    X = pivoted_data.index.values.reshape(-1, 1)  # Years as the independent variable
    y = pivoted_data[technology].values  # Costs as the dependent variable
    
    # Remove NaN values
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Store the slope (coefficient) of the linear regression line
    slopes[technology] = model.coef_[0]

# Print the slopes for each technology sorted in ascending order
for technology, slope in sorted(slopes.items(), key=lambda item: item[1]):
    print(f"The slope for {technology} is {slope:.2f}")

plt.figure(figsize=(10, 6))

# Plotting the data for each technology
pivoted_data.plot(kind='line', marker='o', figsize=(10, 6), ax=plt.gca())

plt.title('Cost Predictions of Technologies Over Time')
plt.xlabel('Year')
plt.ylabel('Cost ($/kW-yr)')
plt.legend(title='Technology', loc='best')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Adjust layout to avoid clipping of labels
plt.show()
