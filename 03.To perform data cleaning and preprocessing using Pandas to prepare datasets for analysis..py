import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("\n" + "=" * 65)
print("AIM 3 : DATA CLEANING AND PREPROCESSING")
print("=" * 65)

data_dirty = {
    'Feature1': [2, 4, np.nan, 8, 10, 120],
    'Feature2': [15, np.nan, 35, 45, 55, 65],
    'Feature3': ['X', 'Y', 'X', 'Z', 'Y', 'X'],
    'Target': [1, 0, 1, 0, 1, 0]
}

df_dirty = pd.DataFrame(data_dirty)
print("\nOriginal Dataset:\n")
print(df_dirty)

print("\nMissing Values Count:\n")
print(df_dirty.isnull().sum())

df_clean = df_dirty.copy()
df_clean['Feature1'].fillna(df_clean['Feature1'].mean(), inplace=True)
df_clean['Feature2'].fillna(df_clean['Feature2'].median(), inplace=True)

print("\nAfter Handling Missing Values:\n")
print(df_clean)

le = LabelEncoder()
df_clean['Feature3_Encoded'] = le.fit_transform(df_clean['Feature3'])

print("\nAfter Label Encoding:\n")
print(df_clean)

scaler = StandardScaler()
columns_to_scale = ['Feature1', 'Feature2']
df_clean[columns_to_scale] = scaler.fit_transform(df_clean[columns_to_scale])

print("\nAfter Feature Scaling:\n")
print(df_clean)
