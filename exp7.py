import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

df = pd.read_csv('../ADS Datasets/travel-times.csv')
# print(df.head(10))

print(df.info())

# Fuel Economy has cells with '-' and empty/nan
df['FuelEconomy'] = df['FuelEconomy'].replace('-', np.nan)
df['FuelEconomy'] = df['FuelEconomy'].astype('float64')
print(df.info())

# Fill the missing values with mean
df['FuelEconomy'] = df['FuelEconomy'].fillna(df['FuelEconomy'].mean())
print(df.head(10))

# Box-plot to visualize the extreme values
sns.boxplot(data=df['FuelEconomy'])
plt.show()

sns.boxplot(data=df['AvgSpeed'])
plt.show()

sns.boxplot(data=df['Distance'])
plt.show()

# Outlier Detection using KNN
X = df[['Distance','AvgSpeed','FuelEconomy']]

# Fit KNN Model
nbrs = NearestNeighbors(n_neighbors=3)
nbrs.fit(X)

# Get distances to the Kth nearest neigbors
dist, index = nbrs.kneighbors(X)
knn_distances = dist[:,-1]

# Set threshold and mark outliers
threshold = np.percentile(knn_distances, 0.95)
df['Outlier'] = (knn_distances > threshold).astype(int)

# Show the outliers
outliers = df[df['Outlier'] == 1]
print(outliers)
print(f'The the total number of outliers are: {outliers.value_counts().sum()}')

# DB-SCAN for outlier detection
dbscan = DBSCAN(eps=1.5, min_samples=5)
df['Outlier-1'] = dbscan.fit_predict(X)

# Outliers are labelled as -1
outliers1 = df[df['Outlier-1'] == -1]
print(outliers1)
print(f"The total number of outliers using DBSCAN: {outliers1.value_counts().sum()}")

# Visualize the clusters
sns.scatterplot(data=df, x='Distance', y='FuelEconomy', hue='Outlier-1', palette={0: 'blue', 1: 'green', -1: 'red'}) # -1 Outlier
plt.show()