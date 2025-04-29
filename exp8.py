import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

df = pd.read_csv('../ADS Datasets/Churn_Modelling.csv')
print(df.info())

# Encode the labels into numerics
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# Scatter Plot of Exited column before SMOTE
sns.scatterplot(data=df, x='CreditScore', y='Age', hue='Exited')
plt.show()

# Features-X , target-Y
X = df.drop('Exited', axis=1)
Y = df['Exited']

# Split the dataset into training and testing set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initiation and Model Fitting for Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)

# Predict the values
y_pred = model.predict(x_test)

# Print the F-1 Score before SMOTE
f1 = f1_score(y_test, y_pred)
print(f'F1-SCORE before SMOTE: {f1}')

# Apply SMOTE
smote_model = SMOTE(random_state=42)
x_res, y_res = smote_model.fit_resample(X, Y)

# Create dataframe with new values generated from SMOTE
res_df = pd.DataFrame(x_res, columns=X.columns)
res_df['Exited'] = y_res

# Scatter-Plot after applying SMOTE
sns.scatterplot(data=res_df, x='CreditScore', y='Age', hue='Exited')
plt.show()

# Use these values to generate the F1-score

X_new = res_df.drop('Exited', axis=1)
Y_new = res_df['Exited']

x_tr, x_te, y_tr, y_te = train_test_split(X_new, Y_new, test_size=0.2, random_state=42)

model_new = DecisionTreeClassifier(max_depth=3, random_state=42)
model_new.fit(x_tr, y_tr)

y_pred_new = model.predict(x_te)

f1_new = f1_score(y_te, y_pred_new)
print(f'New F-1 Score after SMOTE {f1_new}')