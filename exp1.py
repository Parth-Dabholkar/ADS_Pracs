import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('../ADS Datasets/loan_data_set.csv')
print(df.head())
print(df.shape)

# Count of Null Values
print(f"Null values in each column: \n{df.isnull().sum()}")

# Mean, Q1, Q3, IQR, max, min
print(df.describe())

# Median and Mode
print(df.median(numeric_only=True))
df1 = df.mode()
print(df1.loc[0])

# Scatter Plot
sns.scatterplot(data=df, x='LoanAmount', y='ApplicantIncome')
plt.title('Scatterplot Loan Amount v/s Applicant Income')
plt.tight_layout()
plt.show()

# BoxPlot
sns.boxplot(data=df[['LoanAmount', 'ApplicantIncome']])
plt.show()

# Trimmed Mean 10%
trimmed_mean = stats.trim_mean(df['LoanAmount'], 0.10)
print(trimmed_mean)

# Summation of all numeric values
print(df.sum(numeric_only=True))

# Frequency and Cummulative Frequency
print(df['LoanAmount'].value_counts())

# Variance, Correlation, Standard Error Mean, Sum of Squares
print(df.var(numeric_only=True))
print(df[['LoanAmount', 'ApplicantIncome']].corr())
print(df.sem(numeric_only=True))

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
sos = 0
for val in df['LoanAmount']:
    sos = val*val + sos
print(sos)

# Skewness and Kurtosis
skewness = df['LoanAmount'].skew()
kurtosis = df['LoanAmount'].kurtosis()
print(skewness, kurtosis)

#Plot skewness and kurtosis
sns.histplot(data=df['LoanAmount'], kde=True, stat='density', bins=30, color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()

# Plot Correlation Matrix
sns.heatmap(df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']].corr(), cmap='plasma')
plt.show()