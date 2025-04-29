import numpy as np
import pandas as pd
import math

df = pd.read_csv('../ADS Datasets/supermarket_sales - Sheet1.csv')
# print(df.info())

# Z-Test when sample size is greater than 30
df1 = df.sample(n=100)

# Sample size and Population Size
sample_size = len(df1)
pop_size = len(df)

# Sample Mean and Population Mean
sample_mean = df1['Total'].mean()
pop_mean = df['Total'].mean()

# Std for sample and population
std_sample = df1['Total'].std()
std_pop = df['Total'].std()

# Z-SCORE CALCULATION
z_score = (sample_mean - pop_mean)/(std_pop/math.sqrt(sample_size))
print(f"Z-SCORE = {z_score}")

# Z-observered for LOS 5%
z_observed = 1.65

if z_score > z_observed: print("Reject Null Hypothesis")
else: print("We do not reject Null Hypothesis")


# For t-test, sample size should be less than 30
df2 = df.sample(n=28)

n = len(df2)
x_bar = df2['Total'].mean()
meu = df['Total'].mean()
s = df2['Total'].std()

t_score = (x_bar - meu)/(s/math.sqrt(n))
print(f"T-SCORE = {t_score}")
t_obs = 1.7

if t_score > t_obs: print("We reject null hypothesis")
else: print("We do not reject null hypothesis")

