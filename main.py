import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
data = pd.read_csv("data.csv")

X = data[['Level']]
y = data["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial regression with degree 2 (you can change the degree as needed)
poly_features = PolynomialFeatures(degree=1)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Create and train the polynomial regression model
poly_regression_model = LinearRegression()
poly_regression_model.fit(X_train_poly, y_train)

# Predictions on the test set
# Assuming your CSV has columns 'Level' and 'Salary'
X = data[['Level']]
y = data['Salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial regression with degree 2 (you can change the degree as needed)
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Create and train the polynomial regression model
poly_regression_model = LinearRegression()
poly_regression_model.fit(X_train_poly, y_train)

# Predictions on the test set
y_pred = poly_regression_model.predict(X_test_poly)

# Visualize the results
plt.scatter(X.values, y, color='blue', label='Actual Data')
plt.scatter(X_test.values, y_pred, color='red', label='Predicted Data')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Polynomial Regression - Actual vs Predicted')
plt.legend()
plt.show()
