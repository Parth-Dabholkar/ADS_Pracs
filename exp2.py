import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../ADS Datasets/travel-times.csv')
print(df.info())

# Remove '-' and replace it with nan
df['FuelEconomy'] = df['FuelEconomy'].replace('-',np.nan)
print(df['FuelEconomy'].head(10))

# Mean Imputation
df_mean_impute = df.copy()
df_mean_impute['FuelEconomy'] = df_mean_impute['FuelEconomy'].astype('float64')
df_mean_impute['FuelEconomy'] = df_mean_impute['FuelEconomy'].fillna(df_mean_impute['FuelEconomy'].mean())
print(df_mean_impute['FuelEconomy'].head(10))

# Median Imputation
df_median_impute = df.copy()
df_median_impute['FuelEconomy'] = df_median_impute['FuelEconomy'].astype('float64')
df_median_impute['FuelEconomy'] = df_median_impute['FuelEconomy'].fillna(df_median_impute['FuelEconomy'].median())
print(df_median_impute['FuelEconomy'].head(10))

# Mode Imputation
df_mode_impute = df.copy()
df_mode_impute['FuelEconomy'] = df_mode_impute['FuelEconomy'].astype('float64')
df_mode_impute['FuelEconomy'] = df_mode_impute['FuelEconomy'].fillna(df_mode_impute['FuelEconomy'].mode()[0])
print(df_mode_impute['FuelEconomy'].head(10))

# Categorical to Ordinal Imputation
df_category = df.copy()
oe_model = OrdinalEncoder()
results = oe_model.fit_transform(df[['GoingTo']])
results = pd.DataFrame(results)
print(results.head(10))

# Regression Imputation
df_regress = df.copy()
df_regress['FuelEconomy'] = df_regress['FuelEconomy'].replace('-', np.nan)
df_regress['FuelEconomy'] = df_regress['FuelEconomy'].astype('float64')

LR = LinearRegression()
train_data = df_regress.dropna(subset=['FuelEconomy', 'Distance'])
LR.fit(train_data[['Distance']], train_data['FuelEconomy'])

missing_rows = df_regress['FuelEconomy'].isnull()

df_regress.loc[missing_rows, 'FuelEconomy'] = LR.predict(df_regress.loc[missing_rows, ['Distance']])

print(df_regress[['Distance', 'FuelEconomy']])