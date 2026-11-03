# ============================================================
# DATA PREPROCESSING PRACTICAL
# Dataset: train_and_test2.csv
# ============================================================

# -------------------- 1. IMPORT LIBRARIES --------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------- 2. LOAD THE DATASET --------------------
df = pd.read_csv("train_and_test2.csv")
print("Dataset loaded successfully!")
print(df.head())

# -------------------- 3. CHECK THE DATASET --------------------
print("\nShape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns)
print("\nDataset information:")
df.info()

# -------------------- 4. DATA CLEANING --------------------
# Remove duplicate rows
df = df.drop_duplicates()

# Remove columns containing only one distinct value (zero variance)
df = df.loc[:, df.nunique(dropna=False) > 1]

# Rename the target column ('2urvived' is a common quirk in this CSV)
df = df.rename(columns={"2urvived": "Survived"})

print("\nDataset after cleaning:")
print(df.head())
print("\nShape after cleaning:", df.shape)

# -------------------- 5. CHECK MISSING VALUES --------------------
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nTotal missing values:", df.isnull().sum().sum())

# -------------------- 6. HANDLE MISSING VALUES --------------------
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values after handling:")
print(df.isnull().sum())

# -------------------- 7. STATISTICAL ANALYSIS --------------------
print("\nStatistical Summary:")
print(df.describe())

if "Age" in df.columns:
    print("\nMean Age:", df["Age"].mean())
    print("Median Age:", df["Age"].median())

if "Fare" in df.columns:
    print("\nMaximum Fare:", df["Fare"].max())
    print("Minimum Fare:", df["Fare"].min())

# -------------------- 8. PLOT HISTOGRAMS --------------------
cols_to_plot = [col for col in ["Age", "Fare"] if col in df.columns]

if cols_to_plot:
    df[cols_to_plot].hist(bins=20, figsize=(10, 4))
    plt.suptitle("Histograms of Age and Fare")
    plt.tight_layout()
    plt.show()

# -------------------- 9. PLOT BOXPLOTS --------------------
if cols_to_plot:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df[cols_to_plot])
    plt.title("Boxplots of Age and Fare")
    plt.tight_layout()
    plt.show()

# -------------------- 10. SEPARATE FEATURES AND TARGET --------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# -------------------- 11. SPLIT TRAINING AND TESTING DATA --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# -------------------- 12. FEATURE SCALING --------------------
scaler = StandardScaler()

# Fit the scaler on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler parameters to transform testing data (avoids data leakage)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training data (first 5 rows):")
print(X_train_scaled[:5])

print("\nScaled testing data (first 5 rows):")
print(X_test_scaled[:5])

# -------------------- 13. DISPLAY FINAL OUTPUT --------------------
print("\n====================================")
print("DATA PREPROCESSING COMPLETED")
print("====================================")
print("Training samples:", X_train_scaled.shape[0])
print("Testing samples:", X_test_scaled.shape[0])
print("Number of features:", X_train_scaled.shape[1])
print("Missing values handled.")
print("Feature scaling completed.")
