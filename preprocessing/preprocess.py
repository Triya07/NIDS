import pandas as pd
import numpy as np
import os

# ==============================
# 1. Load Dataset
# ==============================

file_path = "dataset/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"

print("Loading dataset...")

df = pd.read_csv(file_path, low_memory=False)

print("Original shape:", df.shape)


# ==============================
# 2. Remove Repeated Headers
# ==============================

df = df[df["Label"] != "Label"].copy()

print("After removing repeated headers:", df.shape)


# ==============================
# 3. Clean Column Names
# ==============================

df.columns = df.columns.str.strip()


# ==============================
# 4. Remove Timestamp
# ==============================

# Timestamp is text and we don't need it
# for our first ML model.

if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])


# ==============================
# 5. Separate Features and Label
# ==============================

X = df.drop(columns=["Label"])
y = df["Label"]


# ==============================
# 6. Convert Features to Numeric
# ==============================

print("\nConverting features to numeric...")

for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")


# ==============================
# 7. Handle Infinite Values
# ==============================

X = X.replace([np.inf, -np.inf], np.nan)


# ==============================
# 8. Remove Rows with Missing Values
# ==============================

before = len(X)

valid_rows = X.notna().all(axis=1)

X = X[valid_rows]
y = y[valid_rows]

after = len(X)

print("Rows removed because of invalid/missing values:", before - after)


# ==============================
# 9. Check Data Types
# ==============================

print("\nFeature data types:")
print(X.dtypes.value_counts())


# ==============================
# 10. Display Labels
# ==============================

print("\nLabels:")
print(y.value_counts())


# ==============================
# 11. Combine Clean Data
# ==============================

clean_df = X.copy()
clean_df["Label"] = y.values


# ==============================
# 12. Save Clean Dataset
# ==============================

os.makedirs("dataset/processed", exist_ok=True)

output_file = "dataset/processed/cleaned_ids2018.csv"

clean_df.to_csv(output_file, index=False)

print("\nClean dataset saved to:")
print(output_file)

print("\nFinal shape:")
print(clean_df.shape)

print("\nPreprocessing completed successfully!")