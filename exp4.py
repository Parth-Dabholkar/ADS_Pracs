from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../ADS Datasets/loan_data_set.csv')
print(df.head())

df = df.dropna()

X = df[['LoanAmount']]
Y = df['ApplicantIncome']

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model initiation and training
LR_model = LinearRegression()
LR_model.fit(x_train, y_train)

# Model Predict
y_pred = LR_model.predict(x_test)

# Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print Metrics
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2)

# Visualization
sns.scatterplot(x=y_test, y=y_pred)
plt.show()


# DECISION TREE

le = LabelEncoder()
df['Self_Employed'] = le.fit_transform(df['Self_Employed'])

X1 = df[['ApplicantIncome']]
Y1 = df['Self_Employed']

x1_train, x1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)

# Initian and training
DecTree = DecisionTreeClassifier(max_depth=3, random_state=42)
DecTree.fit(x1_train, y1_train)

# Predict the values
y_tree_pred = DecTree.predict(x1_test)

conf_matrix = confusion_matrix(y1_test, y_tree_pred)
accuracy = accuracy_score(y1_test, y_tree_pred)
precision = precision_score(y1_test, y_tree_pred)
recall = recall_score(y1_test, y_tree_pred)
f1 = f1_score(y1_test, y_tree_pred)

print("Confusion Matrix:\n", conf_matrix)