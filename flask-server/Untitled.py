#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# Load the dataset
fish_data = pd.read_csv("Fish.csv")

# Display the first few rows of the dataset
print(fish_data.head())

# Check for missing values
print(fish_data.isnull().sum())

# Data Exploration and Visualization
sns.pairplot(fish_data)
plt.show()

# Selecting features and target variable
X = fish_data[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = fish_data['Weight']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Training MAE:", mean_absolute_error(y_train, y_pred_train))
print("Testing MAE:", mean_absolute_error(y_test, y_pred_test))

print("Training MSE:", mean_squared_error(y_train, y_pred_train))
print("Testing MSE:", mean_squared_error(y_test, y_pred_test))

print("Training R2 Score:", r2_score(y_train, y_pred_train))
print("Testing R2 Score:", r2_score(y_test, y_pred_test))

# Save the model
import joblib
joblib.dump(model, 'fish_weight_prediction_model.pkl')
