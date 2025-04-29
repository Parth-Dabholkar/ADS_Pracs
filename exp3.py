import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../ADS Datasets/Iris.csv')
print(df.info())

# Univariate Data Visualizations

# Histogram
sns.histplot(data=df['SepalLengthCm'], bins=10, color='skyblue')
plt.title('Histogram')
plt.show()

# Quartile Plot
sns.boxplot(data=df[['SepalLengthCm', 'PetalLengthCm']], color='green')
plt.title('Box-Plot')
plt.show()

# Distribution Chart: KDE curve
sns.histplot(data=df['SepalLengthCm'], bins=20, kde=True, stat='density', color='blue')
plt.title('Distribution Chart')
plt.show()

# Multivariate Data Visualizatins

# Find the best correlated features for scatterplot
print(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].corr())

# Scatter-Plot
sns.scatterplot(df[['PetalWidthCm', 'PetalLengthCm']], palette='mako')
plt.title('Scatter-Plot')
plt.show()

# Scatter Matrix
sns.pairplot(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], palette='mako')
plt.title('Scatter Matrix')
plt.show()

# Heat-Map
sns.heatmap(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].corr(), cmap='viridis')
plt.title('Heat-Map')
plt.show()