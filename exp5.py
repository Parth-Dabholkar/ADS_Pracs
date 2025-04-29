from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import rand_score, adjusted_rand_score, silhouette_score, normalized_mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../ADS Datasets/loan_data_set.csv')
df = df.dropna()

data = df[['ApplicantIncome']]

kmeans = KMeans(n_clusters=3, random_state=42)

df['Cluster'] = kmeans.fit_predict(data)

cluster_name = {0: 'Low', 1: 'Medium', 2: 'High'}
df['Cluster'] = df['Cluster'].map(cluster_name)

true_labels = []
for income in df['ApplicantIncome']:
    if income <= 4000:
        true_labels.append(0)  # Low income
    elif income <= 8000:
        true_labels.append(1)  # Medium income
    else:
        true_labels.append(2)  # High income

df['TrueLabel'] = true_labels

sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Cluster')
plt.show()

# Metrics
rand_index = rand_score(df['TrueLabel'], df['Cluster'])
adj_rand_index = adjusted_rand_score(df['TrueLabel'], df['Cluster'])
mutual_info = normalized_mutual_info_score(df['TrueLabel'], df['Cluster'])
sil_score = silhouette_score(data, df['Cluster'])

# Results
print("Rand Index:", rand_index)
print("Adjusted Rand Index:", adj_rand_index)
print("Normalized Mutual Information Score:", mutual_info)
print("Silhouette Score:", sil_score)
