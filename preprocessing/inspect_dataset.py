import pandas as pd
import numpy as np

# Dataset path
file_path = "dataset/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"

print("Loading dataset...")

# Load dataset
df = pd.read_csv(file_path, low_memory=False)

print("\nOriginal dataset shape:", df.shape)

# Remove repeated header rows
df = df[df["Label"] != "Label"]

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove rows with missing labels
df = df.dropna(subset=["Label"])

# Remove infinite values
df = df.replace([np.inf, -np.inf], np.nan)

print("\nCleaned dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["Label"].value_counts())

print("\nDataset information:")
print(df.info())